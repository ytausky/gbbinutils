use super::*;
use self::TerminalKind::*;

use std::iter;
use std::marker::PhantomData;

type Lookahead = Option<TerminalKind>;

fn follows_line(lookahead: &Lookahead) -> bool {
    match *lookahead {
        None | Some(Eol) => true,
        _ => false,
    }
}

pub fn parse_src<I, B>(tokens: I, block_context: B)
where
    I: Iterator<Item = Token<B::TokenSpec>>,
    B: BlockContext,
{
    let mut parser = Parser {
        tokens: tokens.peekable(),
        _phantom: PhantomData,
    };
    parser.parse_block(block_context)
}

struct Parser<I: Iterator, B: BlockContext> {
    tokens: iter::Peekable<I>,
    _phantom: PhantomData<B>,
}

impl<I, B> Parser<I, B>
where
    B: BlockContext,
    I: Iterator<Item = Token<B::TokenSpec>>,
{
    fn lookahead(&mut self) -> Lookahead {
        self.tokens.peek().map(|t| t.kind())
    }

    fn bump(&mut self) -> I::Item {
        self.tokens.next().unwrap()
    }

    fn expect(&mut self, expected: &Lookahead) -> I::Item {
        assert_eq!(self.lookahead(), *expected);
        self.bump()
    }

    fn expect_label(&mut self) -> <B::TokenSpec as TokenSpec>::Label {
        match self.tokens.next() {
            Some(Token::Label(label)) => label,
            _ => panic!(),
        }
    }

    fn expect_command(&mut self) -> <B::TokenSpec as TokenSpec>::Command {
        match self.tokens.next() {
            Some(Token::Command(command)) => command,
            _ => panic!(),
        }
    }

    fn expect_atom(&mut self) -> <B::TokenSpec as TokenSpec>::Atom {
        match self.tokens.next() {
            Some(Token::Atom(atom)) => atom,
            _ => panic!(),
        }
    }

    fn parse_block(&mut self, block_context: B) {
        self.parse_list(
            &Some(Eol),
            Option::is_none,
            |p, c| p.parse_line(c),
            block_context,
        );
    }

    fn parse_line(&mut self, block_context: B) -> B {
        if self.lookahead() == Some(Label) {
            self.parse_labeled_line(block_context)
        } else {
            self.parse_unlabeled_line(block_context)
        }
    }

    fn parse_labeled_line(&mut self, mut block_context: B) -> B {
        let label = self.expect_label();
        if self.lookahead() == Some(Colon) {
            self.bump();
        }
        if self.lookahead() == Some(Macro) {
            self.parse_macro_definition(label, block_context)
        } else {
            block_context.add_label(label);
            self.parse_unlabeled_line(block_context)
        }
    }

    fn parse_unlabeled_line(&mut self, block_context: B) -> B {
        match self.lookahead() {
            ref t if follows_line(t) => block_context,
            Some(Command) => self.parse_command(block_context),
            Some(Atom) => self.parse_macro_invocation(block_context),
            _ => panic!(),
        }
    }

    fn parse_macro_definition(
        &mut self,
        label: <B::TokenSpec as TokenSpec>::Label,
        block_context: B,
    ) -> B {
        self.expect(&Some(Macro));
        let mut macro_block_context = block_context.enter_macro_def(label);
        self.expect(&Some(Eol));
        while self.lookahead() != Some(Endm) {
            macro_block_context.push_terminal(self.bump())
        }
        self.expect(&Some(Endm));
        macro_block_context.exit_terminal_seq()
    }

    fn parse_command(&mut self, block_context: B) -> B {
        let first_token = self.expect_command();
        let mut instruction_context = block_context.enter_command(first_token);
        instruction_context = self.parse_argument_list(instruction_context);
        instruction_context.exit_command()
    }

    fn parse_macro_invocation(&mut self, block_context: B) -> B {
        let macro_name = self.expect_atom();
        let mut invocation_context = block_context.enter_macro_invocation(macro_name);
        invocation_context = self.parse_macro_arg_list(invocation_context);
        invocation_context.exit_macro_invocation()
    }

    fn parse_argument_list(&mut self, instruction_context: B::CommandContext) -> B::CommandContext {
        self.parse_list(
            &Some(Comma),
            follows_line,
            |p, c| p.parse_argument(c),
            instruction_context,
        )
    }

    fn parse_macro_arg_list(
        &mut self,
        invocation_context: B::MacroInvocationContext,
    ) -> B::MacroInvocationContext {
        self.parse_list(
            &Some(Comma),
            follows_line,
            |p, c| {
                let mut arg_context = c.enter_macro_arg();
                let mut next_token = p.lookahead();
                while next_token != Some(Comma) && !follows_line(&next_token) {
                    arg_context.push_terminal(p.bump());
                    next_token = p.lookahead()
                }
                arg_context.exit_terminal_seq()
            },
            invocation_context,
        )
    }

    fn parse_list<F, P, C>(
        &mut self,
        delimiter: &Lookahead,
        mut follow: F,
        mut parser: P,
        mut context: C,
    ) -> C
    where
        F: FnMut(&Lookahead) -> bool,
        P: FnMut(&mut Self, C) -> C,
    {
        let first_terminal = self.lookahead();
        if !follow(&first_terminal) {
            context = parser(self, context);
            while self.lookahead() == *delimiter {
                self.bump();
                context = parser(self, context)
            }
            assert!(follow(&self.lookahead()));
        }
        context
    }

    fn parse_argument(&mut self, mut instruction_context: B::CommandContext) -> B::CommandContext {
        let expr = self.parse_expression();
        instruction_context.add_argument(expr);
        instruction_context
    }

    fn parse_expression(&mut self) -> SynExpr<I::Item> {
        if self.lookahead() == Some(OpeningBracket) {
            self.parse_deref_expression()
        } else {
            let token = self.bump();
            SynExpr::from(token)
        }
    }

    fn parse_deref_expression(&mut self) -> SynExpr<I::Item> {
        self.expect(&Some(OpeningBracket));
        let expr = self.parse_expression();
        self.expect(&Some(ClosingBracket));
        expr.deref()
    }
}

#[cfg(test)]
mod tests {
    use super::parse_src;

    use frontend::syntax::{self, Token::{self, *}, TokenSpec};

    #[test]
    fn parse_empty_src() {
        assert_eq_actions(&[], &[])
    }

    struct TestContext {
        actions: Vec<Action>,
        token_seq_kind: Option<TokenSeqKind>,
    }

    impl TestContext {
        fn new() -> TestContext {
            TestContext {
                actions: Vec::new(),
                token_seq_kind: None,
            }
        }
    }

    #[derive(Debug, PartialEq)]
    enum Action {
        AddArgument(TestExpr),
        AddLabel(<TestTokenSpec as TokenSpec>::Label),
        EnterInstruction(<TestTokenSpec as TokenSpec>::Command),
        EnterMacroArg,
        EnterMacroDef(<TestTokenSpec as TokenSpec>::Label),
        EnterMacroInvocation(<TestTokenSpec as TokenSpec>::Atom),
        ExitInstruction,
        ExitMacroArg,
        ExitMacroDef,
        ExitMacroInvocation,
        PushTerminal(TestToken),
    }

    enum TokenSeqKind {
        MacroArg,
        MacroDef,
    }

    #[derive(Clone, Debug, PartialEq)]
    struct TestTokenSpec;

    impl TokenSpec for TestTokenSpec {
        type Atom = usize;
        type Command = usize;
        type Label = usize;
    }

    type TestToken = Token<TestTokenSpec>;
    type TestExpr = syntax::SynExpr<TestToken>;

    impl<'a> syntax::BlockContext for &'a mut TestContext {
        type TokenSpec = TestTokenSpec;
        type CommandContext = Self;
        type MacroDefContext = Self;
        type MacroInvocationContext = Self;

        fn add_label(&mut self, label: <Self::TokenSpec as TokenSpec>::Label) {
            self.actions.push(Action::AddLabel(label))
        }

        fn enter_command(
            self,
            name: <Self::TokenSpec as TokenSpec>::Command,
        ) -> Self::CommandContext {
            self.actions.push(Action::EnterInstruction(name));
            self
        }

        fn enter_macro_def(
            self,
            label: <Self::TokenSpec as TokenSpec>::Label,
        ) -> Self::MacroDefContext {
            self.actions.push(Action::EnterMacroDef(label));
            self.token_seq_kind = Some(TokenSeqKind::MacroDef);
            self
        }

        fn enter_macro_invocation(
            self,
            name: <Self::TokenSpec as TokenSpec>::Atom,
        ) -> Self::MacroInvocationContext {
            self.actions.push(Action::EnterMacroInvocation(name));
            self
        }
    }

    impl<'a> syntax::CommandContext for &'a mut TestContext {
        type Terminal = TestToken;
        type EnclosingContext = Self;

        fn add_argument(&mut self, expr: syntax::SynExpr<Self::Terminal>) {
            self.actions.push(Action::AddArgument(expr))
        }

        fn exit_command(self) -> Self::EnclosingContext {
            self.actions.push(Action::ExitInstruction);
            self
        }
    }

    impl<'a> syntax::MacroInvocationContext for &'a mut TestContext {
        type Terminal = TestToken;
        type EnclosingContext = Self;
        type MacroArgContext = Self;

        fn enter_macro_arg(self) -> Self::MacroArgContext {
            self.actions.push(Action::EnterMacroArg);
            self.token_seq_kind = Some(TokenSeqKind::MacroArg);
            self
        }

        fn exit_macro_invocation(self) -> Self::EnclosingContext {
            self.actions.push(Action::ExitMacroInvocation);
            self
        }
    }

    impl<'a> syntax::TerminalSeqContext for &'a mut TestContext {
        type Terminal = TestToken;
        type EnclosingContext = Self;

        fn push_terminal(&mut self, terminal: Self::Terminal) {
            self.actions.push(Action::PushTerminal(terminal))
        }

        fn exit_terminal_seq(self) -> Self::EnclosingContext {
            self.actions
                .push(match *self.token_seq_kind.as_ref().unwrap() {
                    TokenSeqKind::MacroArg => Action::ExitMacroArg,
                    TokenSeqKind::MacroDef => Action::ExitMacroDef,
                });
            self.token_seq_kind = None;
            self
        }
    }

    #[test]
    fn parse_empty_line() {
        assert_eq_actions(&[Eol], &[])
    }

    fn assert_eq_actions(tokens: &[TestToken], expected_actions: &[Action]) {
        let mut parsing_context = TestContext::new();
        parse_src(tokens.iter().cloned(), &mut parsing_context);
        assert_eq!(parsing_context.actions, expected_actions)
    }

    #[test]
    fn parse_nullary_instruction() {
        assert_eq_actions(&[Command(0)], &inst(0, vec![]))
    }

    #[test]
    fn parse_nullary_instruction_after_eol() {
        assert_eq_actions(&[Eol, Command(1)], &inst(1, vec![]))
    }

    fn inst(name: <TestTokenSpec as TokenSpec>::Command, args: Vec<Vec<Action>>) -> Vec<Action> {
        let mut result = vec![Action::EnterInstruction(name)];
        for mut arg in args {
            result.append(&mut arg);
        }
        result.push(Action::ExitInstruction);
        result
    }

    #[test]
    fn parse_nullary_instruction_followed_by_eol() {
        assert_eq_actions(&[Command(0), Eol], &inst(0, vec![]))
    }

    #[test]
    fn parse_unary_instruction() {
        assert_eq_actions(&[Command(0), Atom(1)], &inst(0, vec![expr(ident(1))]))
    }

    fn expr(expression: TestExpr) -> Vec<Action> {
        vec![Action::AddArgument(expression)]
    }

    fn ident(identifier: <TestTokenSpec as TokenSpec>::Atom) -> TestExpr {
        syntax::SynExpr::Atom(Token::Atom(identifier))
    }

    #[test]
    fn parse_binary_instruction() {
        assert_eq_actions(
            &[Command(0), Atom(1), Comma, Atom(3)],
            &inst(0, vec![expr(ident(1)), expr(ident(3))]),
        );
    }

    #[test]
    fn parse_two_instructions() {
        let tokens = &[
            Command(0),
            Atom(1),
            Comma,
            Atom(3),
            Eol,
            Command(5),
            Atom(6),
            Comma,
            Atom(8),
        ];
        let expected_actions = &concat(vec![
            inst(0, vec![expr(ident(1)), expr(ident(3))]),
            inst(5, vec![expr(ident(6)), expr(ident(8))]),
        ]);
        assert_eq_actions(tokens, expected_actions)
    }

    fn concat(actions: Vec<Vec<Action>>) -> Vec<Action> {
        let mut result = Vec::new();
        for mut vector in actions {
            result.append(&mut vector)
        }
        result
    }

    #[test]
    fn parse_two_instructions_separated_by_blank_line() {
        let tokens = &[
            Command(0),
            Atom(1),
            Comma,
            Atom(3),
            Eol,
            Eol,
            Command(6),
            Atom(7),
            Comma,
            Atom(9),
        ];
        let expected_actions = &concat(vec![
            inst(0, vec![expr(ident(1)), expr(ident(3))]),
            inst(6, vec![expr(ident(7)), expr(ident(9))]),
        ]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_empty_macro_definition() {
        let tokens = &[Label(0), Colon, Macro, Eol, Endm];
        let expected_actions = &macro_def(0, vec![]);
        assert_eq_actions(tokens, expected_actions);
    }

    fn macro_def(
        label: <TestTokenSpec as TokenSpec>::Label,
        tokens: Vec<TestToken>,
    ) -> Vec<Action> {
        let mut result = vec![Action::EnterMacroDef(label)];
        result.extend(tokens.into_iter().map(|t| Action::PushTerminal(t)));
        result.push(Action::ExitMacroDef);
        result
    }

    #[test]
    fn parse_macro_definition_with_instruction() {
        let tokens = &[Label(0), Colon, Macro, Eol, Command(4), Eol, Endm];
        let expected_actions = &macro_def(0, vec![Command(4), Eol]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_label() {
        let tokens = &[Label(0), Eol];
        let expected_actions = &add_label(0, vec![]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_label_colon() {
        let tokens = &[Label(0), Colon, Eol];
        let expected_actions = &add_label(0, vec![]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_labeled_instruction() {
        let tokens = &[Label(0), Colon, Command(2), Eol];
        let expected_actions = &add_label(0, inst(2, vec![]));
        assert_eq_actions(tokens, expected_actions)
    }

    fn add_label(
        label: <TestTokenSpec as TokenSpec>::Label,
        mut following_actions: Vec<Action>,
    ) -> Vec<Action> {
        let mut result = vec![Action::AddLabel(label)];
        result.append(&mut following_actions);
        result
    }

    #[test]
    fn parse_deref_operand() {
        let tokens = &[Command(0), OpeningBracket, Atom(2), ClosingBracket];
        let expected_actions = &inst(0, vec![expr(deref(ident(2)))]);
        assert_eq_actions(tokens, expected_actions)
    }

    fn deref(expr: TestExpr) -> TestExpr {
        syntax::SynExpr::Deref(Box::new(expr))
    }

    #[test]
    fn parse_nullary_macro_invocation() {
        let tokens = &[Atom(0)];
        let expected_actions = &invoke(0, vec![]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_invocation() {
        let tokens = &[Atom(0), Atom(1)];
        let expected_actions = &invoke(0, vec![vec![Atom(1)]]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_invocation_with_multiple_terminals() {
        let tokens = &[Atom(0), Atom(1), Atom(2), Atom(3)];
        let expected_actions = &invoke(0, vec![vec![Atom(1), Atom(2), Atom(3)]]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_binary_macro_invocation_with_multiple_terminals() {
        let tokens = &[Atom(0), Atom(1), Atom(2), Comma, Atom(4), Atom(5), Atom(6)];
        let expected_actions = &invoke(
            0,
            vec![vec![Atom(1), Atom(2)], vec![Atom(4), Atom(5), Atom(6)]],
        );
        assert_eq_actions(tokens, expected_actions)
    }

    fn invoke(name: <TestTokenSpec as TokenSpec>::Atom, args: Vec<Vec<TestToken>>) -> Vec<Action> {
        let mut actions = vec![Action::EnterMacroInvocation(name)];
        for arg in args.into_iter() {
            actions.push(Action::EnterMacroArg);
            actions.extend(arg.into_iter().map(|t| Action::PushTerminal(t)));
            actions.push(Action::ExitMacroArg);
        }
        actions.push(Action::ExitMacroInvocation);
        actions
    }
}
