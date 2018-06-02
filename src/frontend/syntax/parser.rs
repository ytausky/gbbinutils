use super::*;

use std::iter;

#[derive(Clone, Debug, PartialEq)]
pub enum Token<S: TokenSpec> {
    ClosingParenthesis,
    Colon,
    Comma,
    Command(S::Command),
    Endm,
    Eol,
    Ident(S::Ident),
    Literal(S::Literal),
    Macro,
    OpeningParenthesis,
}

impl Copy for Token<()> {}

impl<S: TokenSpec> Token<S> {
    fn kind(&self) -> Token<()> {
        use self::Token::*;
        match *self {
            ClosingParenthesis => ClosingParenthesis,
            Colon => Colon,
            Comma => Comma,
            Command(_) => Command(()),
            Endm => Endm,
            Eol => Eol,
            Ident(_) => Ident(()),
            Literal(_) => Literal(()),
            Macro => Macro,
            OpeningParenthesis => OpeningParenthesis,
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

macro_rules! mk_expect {
    ($name:ident, $ret_ty:ident) => {
        fn $name(&mut self) -> (S::$ret_ty, T) {
            match self.tokens.next() {
                Some((Token::$ret_ty(inner), t)) => (inner, t),
                _ => panic!(),
            }
        }
    }
}

impl<S: TokenSpec, T, I: Iterator<Item = (Token<S>, T)>> Parser<I> {
    mk_expect!(expect_command, Command);
    mk_expect!(expect_ident, Ident);

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

    fn parse_file<F: FileContext<S, T>>(&mut self, actions: F) {
        self.parse_list(
            &Some(Token::Eol),
            Option::is_none,
            |p, c| p.parse_line(c),
            actions,
        );
    }

    fn parse_line<F: FileContext<S, T>>(&mut self, actions: F) -> F {
        if self.lookahead() == Some(Token::Ident(())) {
            self.parse_potentially_labeled_line(actions)
        } else {
            self.parse_unlabeled_line(actions)
        }
    }

    fn parse_potentially_labeled_line<F: FileContext<S, T>>(&mut self, mut actions: F) -> F {
        let ident = self.expect_ident();
        if self.lookahead() == Some(Token::Colon) {
            self.bump();
            actions.add_label(ident);
            self.parse_unlabeled_line(actions)
        } else {
            self.parse_macro_invocation(ident, actions)
        }
    }

    fn parse_unlabeled_line<F: FileContext<S, T>>(&mut self, actions: F) -> F {
        match self.lookahead() {
            t if follows_line(&t) => actions,
            Some(Token::Command(())) => self.parse_command(actions),
            Some(Token::Ident(())) => {
                let ident = self.expect_ident();
                self.parse_macro_invocation(ident, actions)
            }
            Some(Token::Macro) => self.parse_macro_def(actions),
            _ => panic!(),
        }
    }

    fn parse_macro_def<F: FileContext<S, T>>(&mut self, actions: F) -> F {
        self.expect(&Some(Token::Macro));
        let name = self.expect_ident();
        let mut macro_def_context = actions.enter_macro_def(name);
        self.expect(&Some(Token::Eol));
        while self.lookahead() != Some(Token::Endm) {
            macro_def_context.push_token(self.bump())
        }
        self.expect(&Some(Token::Endm));
        macro_def_context.exit()
    }

    fn parse_command<F: FileContext<S, T>>(&mut self, actions: F) -> F {
        let first_token = self.expect_command();
        let mut command_context = actions.enter_command(first_token);
        command_context = self.parse_argument_list(command_context);
        command_context.exit()
    }

    fn parse_macro_invocation<F: FileContext<S, T>>(
        &mut self,
        name: (S::Ident, T),
        actions: F,
    ) -> F {
        let mut invocation_context = actions.enter_macro_invocation(name);
        invocation_context = self.parse_macro_arg_list(invocation_context);
        invocation_context.exit()
    }

    fn parse_argument_list<C: CommandContext<T, TokenSpec = S>>(&mut self, actions: C) -> C {
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
                let mut arg_context = actions.enter_macro_arg();
                let mut next_token = p.lookahead();
                while next_token != Some(Token::Comma) && !follows_line(&next_token) {
                    arg_context.push_token(p.bump());
                    next_token = p.lookahead()
                }
                arg_context.exit()
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

    fn parse_argument<C: CommandContext<T, TokenSpec = S>>(&mut self, mut actions: C) -> C {
        let expr = self.parse_expression();
        actions.add_argument(expr);
        actions
    }

    fn parse_expression(&mut self) -> ParsedExpr<S, T> {
        if self.lookahead() == Some(Token::OpeningParenthesis) {
            self.parse_deref_expression()
        } else {
            self.parse_atomic_expr()
        }
    }

    fn parse_deref_expression(&mut self) -> ParsedExpr<S, T> {
        self.expect(&Some(Token::OpeningParenthesis));
        let expr = self.parse_expression();
        self.expect(&Some(Token::ClosingParenthesis));
        expr.deref()
    }

    fn parse_atomic_expr(&mut self) -> ParsedExpr<S, T> {
        let (token, token_ref) = self.bump();
        match token {
            Token::Ident(ident) => ParsedExpr::Ident((ident, token_ref)),
            Token::Literal(literal) => ParsedExpr::Literal((literal, token_ref)),
            _ => panic!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_src,
                Token::{self, *}};

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
        EnterMacroInvocation(<TestTokenSpec as TokenSpec>::Ident),
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
        type Command = ();
        type Ident = usize;
        type Literal = usize;
    }

    type TestToken = Token<TestTokenSpec>;
    type TestTrackingData = usize;
    type TestExpr = syntax::ParsedExpr<TestTokenSpec, TestTrackingData>;

    impl<'a> syntax::FileContext<TestTokenSpec, TestTrackingData> for &'a mut TestContext {
        type CommandContext = Self;
        type MacroDefContext = Self;
        type MacroInvocationContext = Self;

        fn add_label(&mut self, (_, n): (<TestTokenSpec as TokenSpec>::Ident, TestTrackingData)) {
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
            (_, n): (<TestTokenSpec as TokenSpec>::Ident, TestTrackingData),
        ) -> Self::MacroDefContext {
            self.actions.push(Action::EnterMacroDef(n));
            self.token_seq_kind = Some(TokenSeqKind::MacroDef);
            self
        }

        fn enter_macro_invocation(
            self,
            name: (<TestTokenSpec as TokenSpec>::Ident, TestTrackingData),
        ) -> Self::MacroInvocationContext {
            self.actions.push(Action::EnterMacroInvocation(name.0));
            self
        }
    }

    impl<'a> syntax::CommandContext<TestTrackingData> for &'a mut TestContext {
        type TokenSpec = TestTokenSpec;
        type Parent = Self;

        fn add_argument(&mut self, expr: TestExpr) {
            self.actions.push(Action::AddArgument(expr))
        }

        fn exit(self) -> Self::Parent {
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

        fn exit(self) -> Self::Parent {
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

        fn exit(self) -> Self::Parent {
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
        assert_eq_actions(&[Command(()), Ident(1)], &inst(0, vec![expr(ident(1))]))
    }

    fn expr(expression: TestExpr) -> Vec<Action> {
        vec![Action::AddArgument(expression)]
    }

    fn ident(identifier: <TestTokenSpec as TokenSpec>::Ident) -> TestExpr {
        syntax::ParsedExpr::Ident((identifier, identifier))
    }

    #[test]
    fn parse_binary_instruction() {
        assert_eq_actions(
            &[Command(()), Ident(1), Comma, Literal(3)],
            &inst(0, vec![expr(ident(1)), expr(atom(3))]),
        );
    }

    fn atom(n: <TestTokenSpec as TokenSpec>::Literal) -> TestExpr {
        syntax::ParsedExpr::Literal((n, n))
    }

    #[test]
    fn parse_two_instructions() {
        let tokens = &[
            Command(()),
            Ident(1),
            Comma,
            Literal(3),
            Eol,
            Command(()),
            Literal(6),
            Comma,
            Ident(8),
        ];
        let expected_actions = &concat(vec![
            inst(0, vec![expr(ident(1)), expr(atom(3))]),
            inst(5, vec![expr(atom(6)), expr(ident(8))]),
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
            Literal(1),
            Comma,
            Ident(3),
            Eol,
            Eol,
            Command(()),
            Ident(7),
            Comma,
            Literal(9),
        ];
        let expected_actions = &concat(vec![
            inst(0, vec![expr(atom(1)), expr(ident(3))]),
            inst(6, vec![expr(ident(7)), expr(atom(9))]),
        ]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_empty_macro_definition() {
        let tokens = &[Macro, Ident(1), Eol, Endm];
        let expected_actions = &macro_def(1, vec![]);
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
        let tokens = &[Macro, Ident(1), Eol, Command(()), Eol, Endm];
        let expected_actions = &macro_def(1, vec![Command(()), Eol]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_label() {
        let tokens = &[Ident(0), Colon, Eol];
        let expected_actions = &add_label((0, 0), vec![]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_labeled_instruction() {
        let tokens = &[Ident(0), Colon, Command(()), Eol];
        let expected_actions = &add_label((0, 0), inst(2, vec![]));
        assert_eq_actions(tokens, expected_actions)
    }

    fn add_label(
        (_, n): (<TestTokenSpec as TokenSpec>::Ident, TestTrackingData),
        mut following_actions: Vec<Action>,
    ) -> Vec<Action> {
        let mut result = vec![Action::AddLabel(n)];
        result.append(&mut following_actions);
        result
    }

    #[test]
    fn parse_deref_operand() {
        let tokens = &[
            Command(()),
            OpeningParenthesis,
            Literal(2),
            ClosingParenthesis,
        ];
        let expected_actions = &inst(0, vec![expr(deref(atom(2)))]);
        assert_eq_actions(tokens, expected_actions)
    }

    fn deref(expr: TestExpr) -> TestExpr {
        syntax::ParsedExpr::Deref(Box::new(expr))
    }

    #[test]
    fn parse_nullary_macro_invocation() {
        let tokens = &[Ident(0)];
        let expected_actions = &invoke(0, vec![]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_invocation() {
        let tokens = &[Ident(0), Literal(1)];
        let expected_actions = &invoke(0, vec![vec![Literal(1)]]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_invocation_with_multiple_terminals() {
        let tokens = &[Ident(0), Literal(1), Literal(2), Literal(3)];
        let expected_actions = &invoke(0, vec![vec![Literal(1), Literal(2), Literal(3)]]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_binary_macro_invocation_with_multiple_terminals() {
        let tokens = &[
            Ident(0),
            Literal(1),
            Literal(2),
            Comma,
            Literal(4),
            Literal(5),
            Literal(6),
        ];
        let expected_actions = &invoke(
            0,
            vec![
                vec![Literal(1), Literal(2)],
                vec![Literal(4), Literal(5), Literal(6)],
            ],
        );
        assert_eq_actions(tokens, expected_actions)
    }

    fn invoke(
        name: <TestTokenSpec as TokenSpec>::Literal,
        args: Vec<Vec<TestToken>>,
    ) -> Vec<Action> {
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
