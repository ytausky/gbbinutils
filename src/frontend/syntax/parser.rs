use super::*;
use diagnostics::SourceRange;

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

fn follows_line(lookahead: Lookahead) -> bool {
    match lookahead {
        None | Some(Token::Eol) => true,
        _ => false,
    }
}

pub fn parse_src<S: TokenSpec, T: SourceRange, I, F>(tokens: I, actions: F)
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

impl<S: TokenSpec, T: SourceRange, I: Iterator<Item = (Token<S>, T)>> Parser<I> {
    mk_expect!(expect_command, Command);
    mk_expect!(expect_ident, Ident);

    fn lookahead(&mut self) -> Lookahead {
        self.tokens.peek().map(|&(ref t, _)| t.kind())
    }

    fn bump(&mut self) -> I::Item {
        self.tokens.next().unwrap()
    }

    fn expect(&mut self, expected: Lookahead) -> I::Item {
        assert_eq!(self.lookahead(), expected);
        self.bump()
    }

    fn parse_file<F: FileContext<S, T>>(&mut self, actions: F) {
        self.parse_list(
            Some(Token::Eol),
            |l| l.is_none(),
            |p, c| p.parse_line(c),
            actions,
        );
    }

    fn parse_line<F: FileContext<S, T>>(&mut self, actions: F) -> F {
        if self.lookahead() == Some(Token::Ident(())) {
            self.parse_potentially_labeled_line(actions)
        } else {
            self.parse_unlabeled_line(actions.enter_line(None)).exit()
        }
    }

    fn parse_potentially_labeled_line<F: FileContext<S, T>>(&mut self, actions: F) -> F {
        let ident = self.expect_ident();
        if self.lookahead() == Some(Token::Colon) {
            self.bump();
            self.parse_unlabeled_line(actions.enter_line(Some(ident)))
        } else {
            self.parse_macro_invocation(ident, actions.enter_line(None))
        }.exit()
    }

    fn parse_unlabeled_line<LA: LineActions<S, T>>(&mut self, actions: LA) -> LA {
        match self.lookahead() {
            t if follows_line(t) => actions,
            Some(Token::Command(())) => self.parse_command(actions),
            Some(Token::Ident(())) => {
                let ident = self.expect_ident();
                self.parse_macro_invocation(ident, actions)
            }
            Some(Token::Macro) => self.parse_macro_def(actions),
            _ => panic!(),
        }
    }

    fn parse_macro_def<LA: LineActions<S, T>>(&mut self, actions: LA) -> LA {
        self.expect(Some(Token::Macro));
        let mut macro_body_actions = self.parse_list(
            Some(Token::Comma),
            follows_line,
            |p, a| p.parse_macro_param(a),
            actions.enter_macro_def(),
        ).exit();
        self.expect(Some(Token::Eol));
        while self.lookahead() != Some(Token::Endm) {
            macro_body_actions.push_token(self.bump())
        }
        self.expect(Some(Token::Endm));
        macro_body_actions.exit()
    }

    fn parse_macro_param<MPA>(&mut self, mut actions: MPA) -> MPA
    where
        MPA: MacroParamsActions<T, TokenSpec = S>,
    {
        actions.add_parameter(self.expect_ident());
        actions
    }

    fn parse_command<LA: LineActions<S, T>>(&mut self, actions: LA) -> LA {
        let first_token = self.expect_command();
        let mut command_context = actions.enter_command(first_token);
        command_context = self.parse_argument_list(command_context);
        command_context.exit()
    }

    fn parse_macro_invocation<LA: LineActions<S, T>>(
        &mut self,
        name: (S::Ident, T),
        actions: LA,
    ) -> LA {
        let mut invocation_context = actions.enter_macro_invocation(name);
        invocation_context = self.parse_macro_arg_list(invocation_context);
        invocation_context.exit()
    }

    fn parse_argument_list<C: CommandContext<T, TokenSpec = S>>(&mut self, actions: C) -> C {
        self.parse_list(
            Some(Token::Comma),
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
            Some(Token::Comma),
            follows_line,
            |p, actions| {
                let mut arg_context = actions.enter_macro_arg();
                let mut next_token = p.lookahead();
                while next_token != Some(Token::Comma) && !follows_line(next_token) {
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
        delimiter: Lookahead,
        mut follow: FP,
        mut parser: P,
        mut context: C,
    ) -> C
    where
        FP: FnMut(Lookahead) -> bool,
        P: FnMut(&mut Self, C) -> C,
    {
        let first_terminal = self.lookahead();
        if !follow(first_terminal) {
            context = parser(self, context);
            while self.lookahead() == delimiter {
                self.bump();
                context = parser(self, context)
            }
            assert!(follow(self.lookahead()));
        }
        context
    }

    fn parse_argument<C: CommandContext<T, TokenSpec = S>>(&mut self, actions: C) -> C {
        self.parse_expression(actions.add_argument()).exit()
    }

    fn parse_expression<EA: ExprActions<T, TokenSpec = S>>(&mut self, actions: EA) -> EA {
        if self.lookahead() == Some(Token::OpeningParenthesis) {
            self.parse_parenthesized_expression(actions)
        } else {
            self.parse_atomic_expr(actions)
        }
    }

    fn parse_parenthesized_expression<EA: ExprActions<T, TokenSpec = S>>(
        &mut self,
        actions: EA,
    ) -> EA {
        let (_, left) = self.expect(Some(Token::OpeningParenthesis));
        let mut actions = self.parse_expression(actions);
        let (_, right) = self.expect(Some(Token::ClosingParenthesis));
        actions.apply_operator((ExprOperator::Parentheses, left.extend(&right)));
        actions
    }

    fn parse_atomic_expr<EA: ExprActions<T, TokenSpec = S>>(&mut self, mut actions: EA) -> EA {
        let (token, interval) = self.bump();
        actions.push_atom((
            match token {
                Token::Ident(ident) => ExprAtom::Ident(ident),
                Token::Literal(literal) => ExprAtom::Literal(literal),
                _ => panic!(),
            },
            interval,
        ));
        actions
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_src, Token::{self, *},
    };

    use diagnostics::SourceRange;
    use frontend::syntax::{self, ExprAtom, ExprNode, ExprOperator, ParsedExpr, TokenSpec};
    use std::borrow::Borrow;
    use std::collections::HashMap;
    use std::fmt::Debug;

    #[derive(Debug, PartialEq)]
    struct Symbolic;

    impl TokenSpec for Symbolic {
        type Command = SymCommand;
        type Ident = SymIdent;
        type Literal = SymLiteral;
    }

    #[derive(Debug, PartialEq)]
    struct SymCommand(usize);

    #[derive(Debug, PartialEq)]
    struct SymIdent(usize);

    #[derive(Debug, PartialEq)]
    struct SymLiteral(usize);

    type SymToken = Token<Symbolic>;

    fn mk_sym_token(id: usize, token: Token<()>) -> SymToken {
        match token {
            Command(()) => Command(SymCommand(id)),
            Ident(()) => Ident(SymIdent(id)),
            Literal(()) => Literal(SymLiteral(id)),
            ClosingParenthesis => ClosingParenthesis,
            Colon => Colon,
            Comma => Comma,
            Endm => Endm,
            Eol => Eol,
            Macro => Macro,
            OpeningParenthesis => OpeningParenthesis,
        }
    }

    struct InputTokens {
        tokens: Vec<SymToken>,
        names: HashMap<String, usize>,
    }

    macro_rules! add_token {
        ($input:expr, $token:expr) => {
            let id = $input.tokens.len();
            $input.tokens.push(mk_sym_token(id, $token))
        };
        ($input:expr, $name:ident @ $token:expr) => {
            $input
                .names
                .insert(stringify!($name).into(), $input.tokens.len());
            add_token!($input, $token)
        };
    }

    macro_rules! input_tokens_impl {
        ($input:expr, ) => {};
        ($input:expr, $token:expr) => {
            add_token!($input, $token)
        };
        ($input:expr, $token:expr, $($tail:tt)*) => {
            add_token!($input, $token);
            input_tokens_impl![$input, $($tail)*]
        };
        ($input:expr, $name:ident @ $token:expr) => {
            add_token!($input, $name @ $token)
        };
        ($input:expr, $name:ident @ $token:expr, $($tail:tt)*) => {
            add_token!($input, $name @ $token);
            input_tokens_impl![$input, $($tail)*]
        }
    }

    macro_rules! input_tokens {
        () => {
            InputTokens {
                tokens: Vec::new(),
                names: HashMap::new(),
            }
        };
        ($($tokens:tt)*) => {{
            let mut input = input_tokens![];
            input_tokens_impl!(input, $($tokens)*);
            input
        }};
    }

    #[test]
    fn test_token_macro() {
        let tokens = input_tokens![
            my_tok @ Command(()),
            Literal(()),
            next_one @ Macro,
        ];
        assert_eq!(
            tokens.tokens,
            [Command(SymCommand(0)), Literal(SymLiteral(1)), Macro]
        );
        assert_eq!(tokens.names.get("my_tok"), Some(&0));
        assert_eq!(tokens.names.get("next_one"), Some(&2))
    }

    #[derive(Clone, Debug, PartialEq)]
    struct SymRange<T> {
        start: T,
        end: T,
    }

    impl<T: Clone> From<T> for SymRange<T> {
        fn from(x: T) -> SymRange<T> {
            SymRange {
                start: x.clone(),
                end: x,
            }
        }
    }

    impl<T: Clone + Debug> SourceRange for SymRange<T> {
        fn extend(&self, other: &Self) -> Self {
            SymRange {
                start: self.start.clone(),
                end: other.end.clone(),
            }
        }
    }

    #[test]
    fn parse_empty_src() {
        assert_eq_actions(input_tokens![], vec![])
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
        AddParameter(SymIdent),
        ApplyExprOperator(ExprOperator),
        EnterArgument,
        EnterInstruction(SymCommand),
        EnterLine(Option<SymIdent>),
        EnterMacroArg,
        EnterMacroBody,
        EnterMacroDef,
        EnterMacroInvocation(SymIdent),
        ExitArgument,
        ExitInstruction,
        ExitLine,
        ExitMacroArg,
        ExitMacroDef,
        ExitMacroInvocation,
        PushExprAtom(ExprAtom<Symbolic>),
        PushTerminal(SymToken),
    }

    enum TokenSeqKind {
        MacroArg,
        MacroDef,
    }

    type TestExpr = syntax::ParsedExpr<Symbolic, SymRange<usize>>;

    impl<'a> syntax::FileContext<Symbolic, SymRange<usize>> for &'a mut TestContext {
        type LineActions = Self;

        fn enter_line(self, label: Option<(SymIdent, SymRange<usize>)>) -> Self::LineActions {
            self.actions
                .push(Action::EnterLine(label.map(|(ident, _)| ident)));
            self
        }
    }

    impl<'a> syntax::LineActions<Symbolic, SymRange<usize>> for &'a mut TestContext {
        type CommandContext = Self;
        type MacroParamsActions = Self;
        type MacroInvocationContext = Self;
        type Parent = Self;

        fn enter_command(
            self,
            (command, _): (SymCommand, SymRange<usize>),
        ) -> Self::CommandContext {
            self.actions.push(Action::EnterInstruction(command));
            self
        }

        fn enter_macro_def(self) -> Self::MacroParamsActions {
            self.actions.push(Action::EnterMacroDef);
            self.token_seq_kind = Some(TokenSeqKind::MacroDef);
            self
        }

        fn enter_macro_invocation(
            self,
            name: (SymIdent, SymRange<usize>),
        ) -> Self::MacroInvocationContext {
            self.actions.push(Action::EnterMacroInvocation(name.0));
            self
        }

        fn exit(self) -> Self::Parent {
            self.actions.push(Action::ExitLine);
            self
        }
    }

    impl<'a> syntax::CommandContext<SymRange<usize>> for &'a mut TestContext {
        type TokenSpec = Symbolic;
        type ArgActions = Self;
        type Parent = Self;

        fn add_argument(self) -> Self::ArgActions {
            self.actions.push(Action::EnterArgument);
            self
        }

        fn exit(self) -> Self::Parent {
            self.actions.push(Action::ExitInstruction);
            self
        }
    }

    impl<'a> syntax::ExprActions<SymRange<usize>> for &'a mut TestContext {
        type TokenSpec = Symbolic;
        type Parent = Self;

        fn push_atom(&mut self, atom: (ExprAtom<Symbolic>, SymRange<usize>)) {
            self.actions.push(Action::PushExprAtom(atom.0))
        }

        fn apply_operator(&mut self, operator: (ExprOperator, SymRange<usize>)) {
            self.actions.push(Action::ApplyExprOperator(operator.0))
        }

        fn exit(self) -> Self::Parent {
            self.actions.push(Action::ExitArgument);
            self
        }
    }

    impl<'a> syntax::MacroParamsActions<SymRange<usize>> for &'a mut TestContext {
        type TokenSpec = Symbolic;
        type MacroBodyActions = Self;
        type Parent = Self;

        fn add_parameter(&mut self, (ident, _): (SymIdent, SymRange<usize>)) {
            self.actions.push(Action::AddParameter(ident))
        }

        fn exit(self) -> Self::MacroBodyActions {
            self.actions.push(Action::EnterMacroBody);
            self
        }
    }

    impl<'a> syntax::MacroInvocationContext<SymRange<usize>> for &'a mut TestContext {
        type Token = SymToken;
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

    impl<'a> syntax::TokenSeqContext<SymRange<usize>> for &'a mut TestContext {
        type Token = SymToken;
        type Parent = Self;

        fn push_token(&mut self, token: (Self::Token, SymRange<usize>)) {
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
        assert_eq_actions(input_tokens![Eol], concat(vec![line(vec![]), line(vec![])]))
    }

    fn assert_eq_actions(tokens: InputTokens, expected_actions: impl IntoIterator<Item = Action>) {
        let mut parsing_context = TestContext::new();
        parse_src(
            tokens
                .tokens
                .into_iter()
                .zip((0..).map(|n| SymRange::from(n))),
            &mut parsing_context,
        );
        assert_eq!(
            parsing_context.actions,
            expected_actions.into_iter().collect::<Vec<_>>()
        )
    }

    fn line(mut actions: Vec<Action>) -> Vec<Action> {
        let mut result = vec![Action::EnterLine(None)];
        result.append(&mut actions);
        result.push(Action::ExitLine);
        result
    }

    #[test]
    fn parse_nullary_instruction() {
        assert_eq_actions(input_tokens![Command(())], line(inst(0, vec![])))
    }

    #[test]
    fn parse_nullary_instruction_after_eol() {
        assert_eq_actions(
            input_tokens![Eol, Command(())],
            concat(vec![line(vec![]), line(inst(1, vec![]))]),
        )
    }

    fn inst(id: usize, args: Vec<Vec<Action>>) -> Vec<Action> {
        let mut result = vec![Action::EnterInstruction(SymCommand(id))];
        for mut arg in args {
            result.push(Action::EnterArgument);
            result.append(&mut arg);
            result.push(Action::ExitArgument);
        }
        result.push(Action::ExitInstruction);
        result
    }

    #[test]
    fn parse_nullary_instruction_followed_by_eol() {
        assert_eq_actions(
            input_tokens![Command(()), Eol],
            concat(vec![line(inst(0, vec![])), line(vec![])]),
        )
    }

    #[test]
    fn parse_unary_instruction() {
        assert_eq_actions(
            input_tokens![Command(()), Ident(())],
            line(inst(0, vec![expr(ident(1))])),
        )
    }

    fn expr(expression: TestExpr) -> Vec<Action> {
        match expression.node {
            ExprNode::Ident(ident) => vec![Action::PushExprAtom(ExprAtom::Ident(ident))],
            ExprNode::Literal(literal) => vec![Action::PushExprAtom(ExprAtom::Literal(literal))],
            ExprNode::Parenthesized(e) => {
                let mut actions = expr(*e);
                actions.push(Action::ApplyExprOperator(ExprOperator::Parentheses));
                actions
            }
        }
    }

    fn ident(id: usize) -> TestExpr {
        ParsedExpr {
            node: ExprNode::Ident(SymIdent(id)),
            interval: id.into(),
        }
    }

    #[test]
    fn parse_binary_instruction() {
        assert_eq_actions(
            input_tokens![Command(()), Ident(()), Comma, Literal(())],
            line(inst(0, vec![expr(ident(1)), expr(atom(3))])),
        );
    }

    fn atom(id: usize) -> TestExpr {
        ParsedExpr {
            node: ExprNode::Literal(SymLiteral(id)),
            interval: id.into(),
        }
    }

    #[test]
    fn parse_two_instructions() {
        let tokens = input_tokens![
            Command(()),
            Ident(()),
            Comma,
            Literal(()),
            Eol,
            Command(()),
            Literal(()),
            Comma,
            Ident(()),
        ];
        let expected_actions = concat(vec![
            line(inst(0, vec![expr(ident(1)), expr(atom(3))])),
            line(inst(5, vec![expr(atom(6)), expr(ident(8))])),
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
        let tokens = input_tokens![
            Command(()),
            Literal(()),
            Comma,
            Ident(()),
            Eol,
            Eol,
            Command(()),
            Ident(()),
            Comma,
            Literal(()),
        ];
        let expected_actions = concat(vec![
            line(inst(0, vec![expr(atom(1)), expr(ident(3))])),
            line(vec![]),
            line(inst(6, vec![expr(ident(7)), expr(atom(9))])),
        ]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_empty_macro_definition() {
        let tokens = input_tokens![Ident(()), Colon, Macro, Eol, Endm];
        let expected_actions = macro_def(0, [], vec![]);
        assert_eq_actions(tokens, expected_actions);
    }

    fn macro_def(label: usize, params: impl Borrow<[usize]>, tokens: Vec<SymToken>) -> Vec<Action> {
        let mut result = vec![
            Action::EnterLine(Some(SymIdent(label))),
            Action::EnterMacroDef,
        ];
        result.extend(
            params
                .borrow()
                .iter()
                .map(|&id| Action::AddParameter(SymIdent(id))),
        );
        result.push(Action::EnterMacroBody);
        result.extend(tokens.into_iter().map(|t| Action::PushTerminal(t)));
        result.push(Action::ExitMacroDef);
        result.push(Action::ExitLine);
        result
    }

    #[test]
    fn parse_macro_definition_with_instruction() {
        let tokens = input_tokens![Ident(()), Colon, Macro, Eol, Command(()), Eol, Endm];
        let expected_actions = macro_def(0, [], vec![Command(SymCommand(4)), Eol]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_nonempty_macro_def_with_two_params() {
        let tokens = input_tokens![
            Ident(()),
            Colon,
            Macro,
            Ident(()),
            Comma,
            Ident(()),
            Eol,
            Command(()),
            Eol,
            Endm,
        ];
        let expected = macro_def(0, vec![3, 5], vec![Command(SymCommand(7)), Eol]);
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_label() {
        let tokens = input_tokens![Ident(()), Colon, Eol];
        let expected_actions = concat(vec![add_label(0, vec![]), line(vec![])]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_labeled_instruction() {
        let tokens = input_tokens![Ident(()), Colon, Command(()), Eol];
        let expected_actions = concat(vec![add_label(0, inst(2, vec![])), line(vec![])]);
        assert_eq_actions(tokens, expected_actions)
    }

    fn add_label(id: usize, mut following_actions: Vec<Action>) -> Vec<Action> {
        let mut result = vec![Action::EnterLine(Some(SymIdent(id)))];
        result.append(&mut following_actions);
        result.push(Action::ExitLine);
        result
    }

    #[test]
    fn parse_deref_operand() {
        let tokens = input_tokens![
            Command(()),
            OpeningParenthesis,
            Literal(()),
            ClosingParenthesis,
        ];
        let expected_actions = line(inst(0, vec![expr(deref(1, atom(2), 3))]));
        assert_eq_actions(tokens, expected_actions)
    }

    fn deref(left: usize, expr: TestExpr, right: usize) -> TestExpr {
        ParsedExpr {
            node: ExprNode::Parenthesized(Box::new(expr)),
            interval: SymRange::from(left).extend(&right.into()),
        }
    }

    #[test]
    fn parse_nullary_macro_invocation() {
        let tokens = input_tokens![Ident(())];
        let expected_actions = line(invoke(0, vec![]));
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_invocation() {
        let tokens = input_tokens![Ident(()), Literal(())];
        let expected_actions = line(invoke(0, vec![vec![Literal(SymLiteral(1))]]));
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_invocation_with_multiple_terminals() {
        let tokens = input_tokens![Ident(()), Literal(()), Literal(()), Literal(())];
        let expected_actions = line(invoke(
            0,
            vec![vec![
                Literal(SymLiteral(1)),
                Literal(SymLiteral(2)),
                Literal(SymLiteral(3)),
            ]],
        ));
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_binary_macro_invocation_with_multiple_terminals() {
        let tokens = input_tokens![
            Ident(()),
            Literal(()),
            Literal(()),
            Comma,
            Literal(()),
            Literal(()),
            Literal(()),
        ];
        let expected_actions = line(invoke(
            0,
            vec![
                vec![Literal(SymLiteral(1)), Literal(SymLiteral(2))],
                vec![
                    Literal(SymLiteral(4)),
                    Literal(SymLiteral(5)),
                    Literal(SymLiteral(6)),
                ],
            ],
        ));
        assert_eq_actions(tokens, expected_actions)
    }

    fn invoke(name: usize, args: Vec<Vec<SymToken>>) -> Vec<Action> {
        let mut actions = vec![Action::EnterMacroInvocation(SymIdent(name))];
        for arg in args.into_iter() {
            actions.push(Action::EnterMacroArg);
            actions.extend(arg.into_iter().map(|t| Action::PushTerminal(t)));
            actions.push(Action::ExitMacroArg);
        }
        actions.push(Action::ExitMacroInvocation);
        actions
    }
}
