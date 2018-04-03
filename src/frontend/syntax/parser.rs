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

pub fn parse_src<I, B>(tokens: I, block_context: &mut B)
where
    I: Iterator<Item = B::Terminal>,
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
    I: Iterator<Item = B::Terminal>,
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

    fn parse_block(&mut self, block_context: &mut B) {
        self.parse_list(&Some(Eol), Option::is_none, |p| p.parse_line(block_context))
    }

    fn parse_line(&mut self, block_context: &mut B) {
        if self.lookahead() == Some(Label) {
            self.parse_labeled_line(block_context)
        } else {
            self.parse_unlabeled_line(block_context)
        }
    }

    fn parse_labeled_line(&mut self, block_context: &mut B) {
        let label = self.expect(&Some(Label));
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

    fn parse_unlabeled_line(&mut self, block_context: &mut B) {
        match self.lookahead() {
            ref t if follows_line(t) => (),
            Some(Command) => self.parse_command(block_context),
            Some(Atom) => self.parse_macro_invocation(block_context),
            _ => panic!(),
        }
    }

    fn parse_macro_definition(&mut self, label: I::Item, block_context: &mut B) {
        self.expect(&Some(Macro));
        let macro_block_context = block_context.enter_macro_definition(label);
        self.expect(&Some(Eol));
        while self.lookahead() != Some(Endm) {
            macro_block_context.push_terminal(self.bump())
        }
        self.expect(&Some(Endm));
        macro_block_context.exit_terminal_sequence()
    }

    fn parse_command(&mut self, block_context: &mut B) {
        let first_token = self.expect(&Some(Command));
        let instruction_context = block_context.enter_command(first_token);
        self.parse_argument_list(instruction_context);
        instruction_context.exit_command()
    }

    fn parse_macro_invocation(&mut self, block_context: &mut B) {
        let macro_name = self.expect(&Some(Atom));
        let invocation_context = block_context.enter_macro_invocation(macro_name);
        self.parse_macro_arg_list(invocation_context);
        invocation_context.exit_macro_invocation()
    }

    fn parse_argument_list(&mut self, instruction_context: &mut B::CommandContext) {
        self.parse_list(&Some(Comma), follows_line, |p| {
            p.parse_argument(instruction_context)
        })
    }

    fn parse_macro_arg_list(&mut self, invocation_context: &mut B::MacroInvocationContext) {
        self.parse_list(&Some(Comma), follows_line, |p| {
            let arg_context = invocation_context.enter_macro_argument();
            let mut next_token = p.lookahead();
            while next_token != Some(Comma) && !follows_line(&next_token) {
                arg_context.push_terminal(p.bump());
                next_token = p.lookahead()
            }
            arg_context.exit_terminal_sequence()
        })
    }

    fn parse_list<F, P>(&mut self, delimiter: &Lookahead, mut follow: F, mut parser: P)
    where
        F: FnMut(&Lookahead) -> bool,
        P: FnMut(&mut Self),
    {
        let first_terminal = self.lookahead();
        if !follow(&first_terminal) {
            parser(self);
            while self.lookahead() == *delimiter {
                self.bump();
                parser(self)
            }
            assert!(follow(&self.lookahead()))
        }
    }

    fn parse_argument(&mut self, instruction_context: &mut B::CommandContext) {
        let expr = self.parse_expression();
        instruction_context.add_argument(expr)
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

    use frontend::syntax;
    use self::syntax::TerminalKind::*;

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
        AddLabel(TestToken),
        EnterInstruction(TestToken),
        EnterMacroArg,
        EnterMacroDef(TestToken),
        EnterMacroInvocation(TestToken),
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

    type TestToken = (syntax::TerminalKind, usize);
    type TestExpr = syntax::SynExpr<TestToken>;

    impl syntax::Terminal for TestToken {
        fn kind(&self) -> syntax::TerminalKind {
            let (ref terminal_kind, _) = *self;
            terminal_kind.clone()
        }
    }

    impl syntax::BlockContext for TestContext {
        type Terminal = TestToken;
        type CommandContext = Self;
        type MacroInvocationContext = Self;
        type TerminalSequenceContext = Self;

        fn add_label(&mut self, label: Self::Terminal) {
            self.actions.push(Action::AddLabel(label))
        }

        fn enter_command(&mut self, name: Self::Terminal) -> &mut Self::CommandContext {
            self.actions.push(Action::EnterInstruction(name));
            self
        }

        fn enter_macro_definition(
            &mut self,
            label: Self::Terminal,
        ) -> &mut Self::TerminalSequenceContext {
            self.actions.push(Action::EnterMacroDef(label));
            self.token_seq_kind = Some(TokenSeqKind::MacroDef);
            self
        }

        fn enter_macro_invocation(
            &mut self,
            name: Self::Terminal,
        ) -> &mut Self::MacroInvocationContext {
            self.actions.push(Action::EnterMacroInvocation(name));
            self
        }
    }

    impl syntax::CommandContext for TestContext {
        type Terminal = TestToken;

        fn add_argument(&mut self, expr: syntax::SynExpr<Self::Terminal>) {
            self.actions.push(Action::AddArgument(expr))
        }

        fn exit_command(&mut self) {
            self.actions.push(Action::ExitInstruction)
        }
    }

    impl syntax::MacroInvocationContext for TestContext {
        type Terminal = TestToken;
        type TerminalSequenceContext = Self;

        fn enter_macro_argument(&mut self) -> &mut Self::TerminalSequenceContext {
            self.actions.push(Action::EnterMacroArg);
            self.token_seq_kind = Some(TokenSeqKind::MacroArg);
            self
        }

        fn exit_macro_invocation(&mut self) {
            self.actions.push(Action::ExitMacroInvocation)
        }
    }

    impl syntax::TerminalSequenceContext for TestContext {
        type Terminal = TestToken;

        fn push_terminal(&mut self, terminal: Self::Terminal) {
            self.actions.push(Action::PushTerminal(terminal))
        }

        fn exit_terminal_sequence(&mut self) {
            self.actions
                .push(match *self.token_seq_kind.as_ref().unwrap() {
                    TokenSeqKind::MacroArg => Action::ExitMacroArg,
                    TokenSeqKind::MacroDef => Action::ExitMacroDef,
                });
            self.token_seq_kind = None
        }
    }

    #[test]
    fn parse_empty_line() {
        assert_eq_actions(&[(Eol, 0)], &[])
    }

    fn assert_eq_actions(tokens: &[TestToken], expected_actions: &[Action]) {
        let mut parsing_constext = TestContext::new();
        parse_src(tokens.iter().cloned(), &mut parsing_constext);
        assert_eq!(parsing_constext.actions, expected_actions)
    }

    #[test]
    fn parse_nullary_instruction() {
        assert_eq_actions(&[(Command, 0)], &inst((Command, 0), vec![]))
    }

    #[test]
    fn parse_nullary_instruction_after_eol() {
        assert_eq_actions(&[(Eol, 0), (Command, 1)], &inst((Command, 1), vec![]))
    }

    fn inst(name: TestToken, args: Vec<Vec<Action>>) -> Vec<Action> {
        let mut result = vec![Action::EnterInstruction(name)];
        for mut arg in args {
            result.append(&mut arg);
        }
        result.push(Action::ExitInstruction);
        result
    }

    #[test]
    fn parse_nullary_instruction_followed_by_eol() {
        assert_eq_actions(&[(Command, 0), (Eol, 1)], &inst((Command, 0), vec![]))
    }

    #[test]
    fn parse_unary_instruction() {
        assert_eq_actions(
            &[(Command, 0), (Atom, 1)],
            &inst((Command, 0), vec![expr(ident((Atom, 1)))]),
        )
    }

    fn expr(expression: TestExpr) -> Vec<Action> {
        vec![Action::AddArgument(expression)]
    }

    fn ident(identifier: TestToken) -> TestExpr {
        syntax::SynExpr::Atom(identifier)
    }

    #[test]
    fn parse_binary_instruction() {
        assert_eq_actions(
            &[(Command, 0), (Atom, 1), (Comma, 2), (Atom, 3)],
            &inst(
                (Command, 0),
                vec![expr(ident((Atom, 1))), expr(ident((Atom, 3)))],
            ),
        );
    }

    #[test]
    fn parse_two_instructions() {
        let tokens = &[
            (Command, 0),
            (Atom, 1),
            (Comma, 2),
            (Atom, 3),
            (Eol, 4),
            (Command, 5),
            (Atom, 6),
            (Comma, 7),
            (Atom, 8),
        ];
        let expected_actions = &concat(vec![
            inst(
                (Command, 0),
                vec![expr(ident((Atom, 1))), expr(ident((Atom, 3)))],
            ),
            inst(
                (Command, 5),
                vec![expr(ident((Atom, 6))), expr(ident((Atom, 8)))],
            ),
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
            (Command, 0),
            (Atom, 1),
            (Comma, 2),
            (Atom, 3),
            (Eol, 4),
            (Eol, 5),
            (Command, 6),
            (Atom, 7),
            (Comma, 8),
            (Atom, 9),
        ];
        let expected_actions = &concat(vec![
            inst(
                (Command, 0),
                vec![expr(ident((Atom, 1))), expr(ident((Atom, 3)))],
            ),
            inst(
                (Command, 6),
                vec![expr(ident((Atom, 7))), expr(ident((Atom, 9)))],
            ),
        ]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_empty_macro_definition() {
        let tokens = &[(Label, 0), (Colon, 1), (Macro, 2), (Eol, 3), (Endm, 4)];
        let expected_actions = &macro_def((Label, 0), vec![]);
        assert_eq_actions(tokens, expected_actions);
    }

    fn macro_def(label: TestToken, tokens: Vec<TestToken>) -> Vec<Action> {
        let mut result = vec![Action::EnterMacroDef(label)];
        result.extend(tokens.into_iter().map(|t| Action::PushTerminal(t)));
        result.push(Action::ExitMacroDef);
        result
    }

    #[test]
    fn parse_macro_definition_with_instruction() {
        let tokens = &[
            (Label, 0),
            (Colon, 1),
            (Macro, 2),
            (Eol, 3),
            (Command, 4),
            (Eol, 5),
            (Endm, 6),
        ];
        let expected_actions = &macro_def((Label, 0), vec![(Command, 4), (Eol, 5)]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_label() {
        let tokens = &[(Label, 0), (Eol, 1)];
        let expected_actions = &add_label((Label, 0), vec![]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_label_colon() {
        let tokens = &[(Label, 0), (Colon, 1), (Eol, 2)];
        let expected_actions = &add_label((Label, 0), vec![]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_labeled_instruction() {
        let tokens = &[(Label, 0), (Colon, 1), (Command, 2), (Eol, 3)];
        let expected_actions = &add_label((Label, 0), inst((Command, 2), vec![]));
        assert_eq_actions(tokens, expected_actions)
    }

    fn add_label(label: TestToken, mut following_actions: Vec<Action>) -> Vec<Action> {
        let mut result = vec![Action::AddLabel(label)];
        result.append(&mut following_actions);
        result
    }

    #[test]
    fn parse_deref_operand() {
        let tokens = &[
            (Command, 0),
            (OpeningBracket, 1),
            (Atom, 2),
            (ClosingBracket, 3),
        ];
        let expected_actions = &inst((Command, 0), vec![expr(deref(ident((Atom, 2))))]);
        assert_eq_actions(tokens, expected_actions)
    }

    fn deref(expr: TestExpr) -> TestExpr {
        syntax::SynExpr::Deref(Box::new(expr))
    }

    #[test]
    fn parse_nullary_macro_invocation() {
        let tokens = &[(Atom, 0)];
        let expected_actions = &invoke((Atom, 0), vec![]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_invocation() {
        let tokens = &[(Atom, 0), (Atom, 1)];
        let expected_actions = &invoke((Atom, 0), vec![vec![(Atom, 1)]]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_invocation_with_multiple_terminals() {
        let tokens = &[(Atom, 0), (Atom, 1), (Atom, 2), (Atom, 3)];
        let expected_actions = &invoke((Atom, 0), vec![vec![(Atom, 1), (Atom, 2), (Atom, 3)]]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_binary_macro_invocation_with_multiple_terminals() {
        let tokens = &[
            (Atom, 0),
            (Atom, 1),
            (Atom, 2),
            (Comma, 3),
            (Atom, 4),
            (Atom, 5),
            (Atom, 6),
        ];
        let expected_actions = &invoke(
            (Atom, 0),
            vec![
                vec![(Atom, 1), (Atom, 2)],
                vec![(Atom, 4), (Atom, 5), (Atom, 6)],
            ],
        );
        assert_eq_actions(tokens, expected_actions)
    }

    fn invoke(name: TestToken, args: Vec<Vec<TestToken>>) -> Vec<Action> {
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
