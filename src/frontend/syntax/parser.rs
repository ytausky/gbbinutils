use super::*;

use std::iter;

#[derive(Clone, Debug, PartialEq)]
pub enum Token<S: TokenSpec> {
    Atom(S::Atom),
    ClosingBracket,
    Colon,
    Comma,
    Command(S::Command),
    Endm,
    Eol,
    Label(S::Label),
    Macro,
    OpeningBracket,
}

impl Copy for Token<()> {}

impl<S: TokenSpec> Token<S> {
    fn kind(&self) -> Token<()> {
        use self::Token::*;
        match *self {
            Atom(_) => Atom(()),
            ClosingBracket => ClosingBracket,
            Colon => Colon,
            Comma => Comma,
            Command(_) => Command(()),
            Endm => Endm,
            Eol => Eol,
            Label(_) => Label(()),
            Macro => Macro,
            OpeningBracket => OpeningBracket,
        }
    }
}

type Lookahead = Option<Token<()>>;

fn follows_line(lookahead: &Lookahead) -> bool {
    match *lookahead {
        None | Some(Token::Eol) => true,
        _ => false,
    }
}

pub fn parse_src<S: TokenSpec, T, I, F>(tokens: I, actions: F)
where
    I: Iterator<Item = (Token<S>, T)>,
    F: FileContext<S, T>,
{
    let mut parser = Parser {
        tokens: tokens.peekable(),
    };
    parser.parse_file(actions)
}

struct Parser<I: Iterator> {
    tokens: iter::Peekable<I>,
}

impl<S: TokenSpec, T, I: Iterator<Item = (Token<S>, T)>> Parser<I> {
    fn lookahead(&mut self) -> Lookahead {
        self.tokens.peek().map(|&(ref t, _)| t.kind())
    }

    fn bump(&mut self) -> I::Item {
        self.tokens.next().unwrap()
    }

    fn expect(&mut self, expected: &Lookahead) -> I::Item {
        assert_eq!(self.lookahead(), *expected);
        self.bump()
    }

    fn expect_label(&mut self) -> (S::Label, T) {
        match self.tokens.next() {
            Some((Token::Label(label), t)) => (label, t),
            _ => panic!(),
        }
    }

    fn expect_command(&mut self) -> (S::Command, T) {
        match self.tokens.next() {
            Some((Token::Command(command), t)) => (command, t),
            _ => panic!(),
        }
    }

    fn expect_atom(&mut self) -> (S::Atom, T) {
        match self.tokens.next() {
            Some((Token::Atom(atom), t)) => (atom, t),
            _ => panic!(),
        }
    }

    fn parse_file<F: FileContext<S, T>>(&mut self, actions: F) {
        self.parse_list(
            &Some(Token::Eol),
            Option::is_none,
            |p, c| p.parse_line(c),
            actions,
        );
    }

    fn parse_line<F: FileContext<S, T>>(&mut self, actions: F) -> F {
        if self.lookahead() == Some(Token::Label(())) {
            self.parse_labeled_line(actions)
        } else {
            self.parse_unlabeled_line(actions)
        }
    }

    fn parse_labeled_line<F: FileContext<S, T>>(&mut self, mut actions: F) -> F {
        let label = self.expect_label();
        if self.lookahead() == Some(Token::Colon) {
            self.bump();
        }
        if self.lookahead() == Some(Token::Macro) {
            self.parse_macro_definition(label, actions)
        } else {
            actions.add_label(label);
            self.parse_unlabeled_line(actions)
        }
    }

    fn parse_unlabeled_line<F: FileContext<S, T>>(&mut self, actions: F) -> F {
        match self.lookahead() {
            ref t if follows_line(t) => actions,
            Some(Token::Command(())) => self.parse_command(actions),
            Some(Token::Atom(())) => self.parse_macro_invocation(actions),
            _ => panic!(),
        }
    }

    fn parse_macro_definition<F: FileContext<S, T>>(
        &mut self,
        label: (S::Label, T),
        actions: F,
    ) -> F {
        self.expect(&Some(Token::Macro));
        let mut actions = actions.enter_macro_def(label);
        self.expect(&Some(Token::Eol));
        while self.lookahead() != Some(Token::Endm) {
            actions.push_token(self.bump())
        }
        self.expect(&Some(Token::Endm));
        actions.exit_token_seq()
    }

    fn parse_command<F: FileContext<S, T>>(&mut self, actions: F) -> F {
        let first_token = self.expect_command();
        let mut actions = actions.enter_command(first_token);
        actions = self.parse_argument_list(actions);
        actions.exit_command()
    }

    fn parse_macro_invocation<F: FileContext<S, T>>(&mut self, actions: F) -> F {
        let macro_name = self.expect_atom();
        let mut actions = actions.enter_macro_invocation(macro_name);
        actions = self.parse_macro_arg_list(actions);
        actions.exit_macro_invocation()
    }

    fn parse_argument_list<C: CommandContext<Token = Token<S>>>(&mut self, actions: C) -> C {
        self.parse_list(
            &Some(Token::Comma),
            follows_line,
            |p, c| p.parse_argument(c),
            actions,
        )
    }

    fn parse_macro_arg_list<M: MacroInvocationContext<T, Token = Token<S>>>(
        &mut self,
        actions: M,
    ) -> M {
        self.parse_list(
            &Some(Token::Comma),
            follows_line,
            |p, actions| {
                let mut actions = actions.enter_macro_arg();
                let mut next_token = p.lookahead();
                while next_token != Some(Token::Comma) && !follows_line(&next_token) {
                    actions.push_token(p.bump());
                    next_token = p.lookahead()
                }
                actions.exit_token_seq()
            },
            actions,
        )
    }

    fn parse_list<FP, P, C>(
        &mut self,
        delimiter: &Lookahead,
        mut follow: FP,
        mut parser: P,
        mut context: C,
    ) -> C
    where
        FP: FnMut(&Lookahead) -> bool,
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

    fn parse_argument<C: CommandContext<Token = Token<S>>>(&mut self, mut actions: C) -> C {
        let expr = self.parse_expression();
        actions.add_argument(expr);
        actions
    }

    fn parse_expression(&mut self) -> SynExpr<Token<S>> {
        if self.lookahead() == Some(Token::OpeningBracket) {
            self.parse_deref_expression()
        } else {
            let (token, _) = self.bump();
            SynExpr::from(token)
        }
    }

    fn parse_deref_expression(&mut self) -> SynExpr<Token<S>> {
        self.expect(&Some(Token::OpeningBracket));
        let expr = self.parse_expression();
        self.expect(&Some(Token::ClosingBracket));
        expr.deref()
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_src, Token::{self, *}};

    use frontend::syntax::{self, TokenSpec};

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
        AddLabel(TestTrackingData),
        EnterInstruction(TestTrackingData),
        EnterMacroArg,
        EnterMacroDef(TestTrackingData),
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
        type Command = ();
        type Label = ();
    }

    type TestToken = Token<TestTokenSpec>;
    type TestTrackingData = usize;
    type TestExpr = syntax::SynExpr<TestToken>;

    impl<'a> syntax::FileContext<TestTokenSpec, TestTrackingData> for &'a mut TestContext {
        type CommandContext = Self;
        type MacroDefContext = Self;
        type MacroInvocationContext = Self;

        fn add_label(&mut self, (_, n): (<TestTokenSpec as TokenSpec>::Label, TestTrackingData)) {
            self.actions.push(Action::AddLabel(n))
        }

        fn enter_command(
            self,
            (_, n): (<TestTokenSpec as TokenSpec>::Command, TestTrackingData),
        ) -> Self::CommandContext {
            self.actions.push(Action::EnterInstruction(n));
            self
        }

        fn enter_macro_def(
            self,
            (_, n): (<TestTokenSpec as TokenSpec>::Label, TestTrackingData),
        ) -> Self::MacroDefContext {
            self.actions.push(Action::EnterMacroDef(n));
            self.token_seq_kind = Some(TokenSeqKind::MacroDef);
            self
        }

        fn enter_macro_invocation(
            self,
            name: (<TestTokenSpec as TokenSpec>::Atom, TestTrackingData),
        ) -> Self::MacroInvocationContext {
            self.actions.push(Action::EnterMacroInvocation(name.0));
            self
        }
    }

    impl<'a> syntax::CommandContext for &'a mut TestContext {
        type Token = TestToken;
        type Parent = Self;

        fn add_argument(&mut self, expr: syntax::SynExpr<Self::Token>) {
            self.actions.push(Action::AddArgument(expr))
        }

        fn exit_command(self) -> Self::Parent {
            self.actions.push(Action::ExitInstruction);
            self
        }
    }

    impl<'a> syntax::MacroInvocationContext<TestTrackingData> for &'a mut TestContext {
        type Token = TestToken;
        type Parent = Self;
        type MacroArgContext = Self;

        fn enter_macro_arg(self) -> Self::MacroArgContext {
            self.actions.push(Action::EnterMacroArg);
            self.token_seq_kind = Some(TokenSeqKind::MacroArg);
            self
        }

        fn exit_macro_invocation(self) -> Self::Parent {
            self.actions.push(Action::ExitMacroInvocation);
            self
        }
    }

    impl<'a> syntax::TokenSeqContext<TestTrackingData> for &'a mut TestContext {
        type Token = TestToken;
        type Parent = Self;

        fn push_token(&mut self, token: (Self::Token, TestTrackingData)) {
            self.actions.push(Action::PushTerminal(token.0))
        }

        fn exit_token_seq(self) -> Self::Parent {
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
        parse_src(tokens.iter().cloned().zip(0..), &mut parsing_context);
        assert_eq!(parsing_context.actions, expected_actions)
    }

    #[test]
    fn parse_nullary_instruction() {
        assert_eq_actions(&[Command(())], &inst(0, vec![]))
    }

    #[test]
    fn parse_nullary_instruction_after_eol() {
        assert_eq_actions(&[Eol, Command(())], &inst(1, vec![]))
    }

    fn inst(n: TestTrackingData, args: Vec<Vec<Action>>) -> Vec<Action> {
        let mut result = vec![Action::EnterInstruction(n)];
        for mut arg in args {
            result.append(&mut arg);
        }
        result.push(Action::ExitInstruction);
        result
    }

    #[test]
    fn parse_nullary_instruction_followed_by_eol() {
        assert_eq_actions(&[Command(()), Eol], &inst(0, vec![]))
    }

    #[test]
    fn parse_unary_instruction() {
        assert_eq_actions(&[Command(()), Atom(1)], &inst(0, vec![expr(ident(1))]))
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
            &[Command(()), Atom(1), Comma, Atom(3)],
            &inst(0, vec![expr(ident(1)), expr(ident(3))]),
        );
    }

    #[test]
    fn parse_two_instructions() {
        let tokens = &[
            Command(()),
            Atom(1),
            Comma,
            Atom(3),
            Eol,
            Command(()),
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
            Command(()),
            Atom(1),
            Comma,
            Atom(3),
            Eol,
            Eol,
            Command(()),
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
        let tokens = &[Label(()), Colon, Macro, Eol, Endm];
        let expected_actions = &macro_def(0, vec![]);
        assert_eq_actions(tokens, expected_actions);
    }

    fn macro_def(label: TestTrackingData, tokens: Vec<TestToken>) -> Vec<Action> {
        let mut result = vec![Action::EnterMacroDef(label)];
        result.extend(tokens.into_iter().map(|t| Action::PushTerminal(t)));
        result.push(Action::ExitMacroDef);
        result
    }

    #[test]
    fn parse_macro_definition_with_instruction() {
        let tokens = &[Label(()), Colon, Macro, Eol, Command(()), Eol, Endm];
        let expected_actions = &macro_def(0, vec![Command(()), Eol]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_label() {
        let tokens = &[Label(()), Eol];
        let expected_actions = &add_label(((), 0), vec![]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_label_colon() {
        let tokens = &[Label(()), Colon, Eol];
        let expected_actions = &add_label(((), 0), vec![]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_labeled_instruction() {
        let tokens = &[Label(()), Colon, Command(()), Eol];
        let expected_actions = &add_label(((), 0), inst(2, vec![]));
        assert_eq_actions(tokens, expected_actions)
    }

    fn add_label(
        (_, n): (<TestTokenSpec as TokenSpec>::Label, TestTrackingData),
        mut following_actions: Vec<Action>,
    ) -> Vec<Action> {
        let mut result = vec![Action::AddLabel(n)];
        result.append(&mut following_actions);
        result
    }

    #[test]
    fn parse_deref_operand() {
        let tokens = &[Command(()), OpeningBracket, Atom(2), ClosingBracket];
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
