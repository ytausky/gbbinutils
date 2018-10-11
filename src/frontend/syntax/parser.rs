use super::*;
use diagnostics::{InternalDiagnostic, Message};
use span::Span;

use std::iter;

type TokenKind = Token<(), (), ()>;

impl Copy for TokenKind {}

impl<C, I, L> Token<C, I, L> {
    fn kind(&self) -> TokenKind {
        use self::Token::*;
        match *self {
            ClosingParenthesis => ClosingParenthesis,
            Colon => Colon,
            Comma => Comma,
            Command(_) => Command(()),
            Endm => Endm,
            Eof => Eof,
            Eol => Eol,
            Ident(_) => Ident(()),
            Literal(_) => Literal(()),
            Macro => Macro,
            OpeningParenthesis => OpeningParenthesis,
            Plus => Plus,
        }
    }
}

const LINE_FOLLOW_SET: &[TokenKind] = &[Token::Eol, Token::Eof];

pub fn parse_src<Id, C, L, S, I, F>(tokens: I, actions: F)
where
    S: Span,
    I: Iterator<Item = (Token<Id, C, L>, S)>,
    F: FileContext<Id, C, L, S>,
{
    let mut parser = Parser {
        tokens: tokens.peekable(),
    };
    parser.parse_file(actions)
}

struct Parser<I: Iterator> {
    tokens: iter::Peekable<I>,
}

trait TokenStream {
    type Ident;
    type Command;
    type Literal;
    type Span: Span;
    type Iter: Iterator<Item = (Token<Self::Ident, Self::Command, Self::Literal>, Self::Span)>;

    fn tokens(&mut self) -> &mut iter::Peekable<Self::Iter>;

    fn lookahead(&mut self) -> TokenKind {
        self.tokens().peek().unwrap().0.kind()
    }

    fn lookahead_is_in(&mut self, kinds: &[TokenKind]) -> bool {
        let lookahead = self.lookahead();
        kinds.iter().any(|x| *x == lookahead)
    }

    fn consume(&mut self, kind: TokenKind) -> bool {
        self.take_token_if(|x| x == kind, |_| ())
    }

    fn take_token_if<P, F>(&mut self, predicate: P, f: F) -> bool
    where
        P: FnOnce(TokenKind) -> bool,
        F: FnOnce(<Self::Iter as Iterator>::Item),
    {
        if predicate(self.lookahead()) {
            f(self.bump());
            true
        } else {
            false
        }
    }

    fn take_token_while<P, F>(&mut self, predicate: P, mut f: F)
    where
        P: Fn(TokenKind) -> bool,
        F: FnMut(<Self::Iter as Iterator>::Item),
    {
        while self.take_token_if(&predicate, &mut f) {}
    }

    fn bump(&mut self) -> <Self::Iter as Iterator>::Item {
        self.tokens().next().unwrap()
    }

    fn expect(&mut self, expected: TokenKind) -> <Self::Iter as Iterator>::Item {
        assert_eq!(self.lookahead(), expected);
        self.bump()
    }
}

impl<Id, C, L, S: Span, I: Iterator<Item = (Token<Id, C, L>, S)>> TokenStream
    for iter::Peekable<I>
{
    type Ident = Id;
    type Command = C;
    type Literal = L;
    type Span = S;
    type Iter = I;

    fn tokens(&mut self) -> &mut iter::Peekable<Self::Iter> {
        self
    }
}

impl<Id, C, L, S: Span, I: Iterator<Item = (Token<Id, C, L>, S)>> TokenStream for Parser<I> {
    type Ident = Id;
    type Command = C;
    type Literal = L;
    type Span = S;
    type Iter = I;

    fn tokens(&mut self) -> &mut iter::Peekable<Self::Iter> {
        &mut self.tokens
    }
}

macro_rules! mk_expect {
    ($name:ident, $variant:ident, $ret_ty:ident) => {
        fn $name(&mut self) -> ($ret_ty, S) {
            match self.tokens.next() {
                Some((Token::$variant(inner), s)) => (inner, s),
                _ => panic!(),
            }
        }
    }
}

impl<Id, C, L, S: Span, I: Iterator<Item = (Token<Id, C, L>, S)>> Parser<I> {
    mk_expect!(expect_command, Command, C);
    mk_expect!(expect_ident, Ident, Id);

    fn parse_file<F: FileContext<Id, C, L, S>>(&mut self, actions: F) {
        self.parse_terminated_list(Token::Eol, &[Token::Eof], |p, c| p.parse_line(c), actions);
    }

    fn parse_line<F: FileContext<Id, C, L, S>>(&mut self, actions: F) -> F {
        if self.lookahead() == Token::Ident(()) {
            self.parse_potentially_labeled_line(actions)
        } else {
            self.parse_unlabeled_line(actions.enter_line(None)).exit()
        }
    }

    fn parse_potentially_labeled_line<F: FileContext<Id, C, L, S>>(&mut self, actions: F) -> F {
        let ident = self.expect_ident();
        if self.consume(Token::Colon) {
            self.parse_unlabeled_line(actions.enter_line(Some(ident)))
        } else {
            self.parse_macro_invocation(ident, actions.enter_line(None))
        }.exit()
    }

    fn parse_unlabeled_line<LA: LineActions<Id, C, L, S>>(&mut self, mut actions: LA) -> LA {
        match self.lookahead() {
            Token::Eol | Token::Eof => actions,
            Token::Command(()) => self.parse_command(actions),
            Token::Ident(()) => {
                let ident = self.expect_ident();
                self.parse_macro_invocation(ident, actions)
            }
            Token::Macro => self.parse_macro_def(actions),
            _ => {
                let (_, range) = self.bump();
                actions.emit_diagnostic(InternalDiagnostic::new(
                    Message::UnexpectedToken,
                    vec![range.clone()],
                    range,
                ));
                actions
            }
        }
    }

    fn parse_macro_def<LA: LineActions<Id, C, L, S>>(&mut self, actions: LA) -> LA {
        self.expect(Token::Macro);
        let mut actions = self.parse_terminated_list(
            Token::Comma,
            LINE_FOLLOW_SET,
            |p, a| p.parse_macro_param(a),
            actions.enter_macro_def(),
        );
        if self.consume(Token::Eol) {
            let mut body_actions = actions.exit();
            self.take_token_while(
                |x| x != Token::Endm && x != Token::Eof,
                |token| body_actions.push_token(token),
            );
            if self.lookahead() == Token::Endm {
                let endm = self.bump();
                body_actions.push_token((Token::Eof, endm.1));
            } else {
                assert_eq!(self.lookahead(), Token::Eof);
                body_actions.emit_diagnostic(InternalDiagnostic::new(
                    Message::UnexpectedEof,
                    iter::empty(),
                    self.tokens.peek().unwrap().1.clone(),
                ))
            }
            body_actions
        } else {
            assert_eq!(self.lookahead(), Token::Eof);
            actions.emit_diagnostic(InternalDiagnostic::new(
                Message::UnexpectedEof,
                iter::empty(),
                self.tokens.peek().unwrap().1.clone(),
            ));
            actions.exit()
        }.exit()
    }

    fn parse_macro_param<MPA>(&mut self, mut actions: MPA) -> MPA
    where
        MPA: MacroParamsActions<S, Command = C, Ident = Id, Literal = L>,
    {
        actions.add_parameter(self.expect_ident());
        actions
    }

    fn parse_command<LA: LineActions<Id, C, L, S>>(&mut self, actions: LA) -> LA {
        let first_token = self.expect_command();
        let mut command_context = actions.enter_command(first_token);
        command_context = self.parse_argument_list(command_context);
        command_context.exit()
    }

    fn parse_macro_invocation<LA: LineActions<Id, C, L, S>>(
        &mut self,
        name: (Id, S),
        actions: LA,
    ) -> LA {
        let mut invocation_context = actions.enter_macro_invocation(name);
        invocation_context = self.parse_macro_arg_list(invocation_context);
        invocation_context.exit()
    }

    fn parse_argument_list<CC: CommandContext<S, Command = C, Ident = Id, Literal = L>>(
        &mut self,
        actions: CC,
    ) -> CC {
        self.parse_terminated_list(
            Token::Comma,
            LINE_FOLLOW_SET,
            |p, c| p.parse_argument(c),
            actions,
        )
    }

    fn parse_macro_arg_list<M: MacroInvocationContext<S, Token = Token<Id, C, L>>>(
        &mut self,
        actions: M,
    ) -> M {
        self.parse_terminated_list(
            Token::Comma,
            LINE_FOLLOW_SET,
            |p, actions| {
                let mut arg_context = actions.enter_macro_arg();
                p.take_token_while(
                    |x| x != Token::Comma && x != Token::Eol && x != Token::Eof,
                    |token| arg_context.push_token(token),
                );
                arg_context.exit()
            },
            actions,
        )
    }

    fn parse_terminated_list<P, A>(
        &mut self,
        delimiter: TokenKind,
        terminators: &[TokenKind],
        parser: P,
        mut context: A,
    ) -> A
    where
        P: FnMut(&mut Self, A) -> A,
        A: DiagnosticsListener<S>,
    {
        context = self.parse_list(delimiter, terminators, parser, context);
        if !self.lookahead_is_in(terminators) {
            let (_, unexpected_range) = self.bump();
            context.emit_diagnostic(InternalDiagnostic::new(
                Message::UnexpectedToken,
                vec![unexpected_range.clone()],
                unexpected_range,
            ));
            while !self.lookahead_is_in(terminators) {
                self.bump();
            }
        }
        context
    }

    fn parse_list<P, A>(
        &mut self,
        delimiter: TokenKind,
        terminators: &[TokenKind],
        mut parser: P,
        context: A,
    ) -> A
    where
        P: FnMut(&mut Self, A) -> A,
        A: DiagnosticsListener<S>,
    {
        if self.lookahead_is_in(terminators) {
            context
        } else {
            self.parse_nonempty_list(delimiter, &mut parser, context)
        }
    }

    fn parse_nonempty_list<P, A>(
        &mut self,
        delimiter: TokenKind,
        parser: &mut P,
        mut actions: A,
    ) -> A
    where
        P: FnMut(&mut Self, A) -> A,
        A: DiagnosticsListener<S>,
    {
        actions = parser(self, actions);
        while self.consume(delimiter) {
            actions = parser(self, actions)
        }
        actions
    }

    fn parse_argument<CC: CommandContext<S, Command = C, Ident = Id, Literal = L>>(
        &mut self,
        actions: CC,
    ) -> CC {
        ExprParser {
            tokens: self,
            actions: actions.add_argument(),
        }.parse()
    }
}

struct ExprParser<'a, T: 'a, A> {
    tokens: &'a mut T,
    actions: A,
}

impl<'a, T: TokenStream + 'a, A> TokenStream for ExprParser<'a, T, A> {
    type Ident = T::Ident;
    type Command = T::Command;
    type Literal = T::Literal;
    type Span = T::Span;
    type Iter = T::Iter;

    fn tokens(&mut self) -> &mut iter::Peekable<Self::Iter> {
        self.tokens.tokens()
    }
}

impl<'a, T, A> ExprParser<'a, T, A>
where
    T: TokenStream + 'a,
    A: ExprActions<T::Span, Ident = T::Ident, Literal = T::Literal>,
{
    fn parse(mut self) -> A::Parent {
        self.parse_expression();
        self.actions.exit()
    }

    fn parse_expression(&mut self) {
        if self.lookahead() == Token::OpeningParenthesis {
            self.parse_parenthesized_expression()
        } else {
            self.parse_infix_expr()
        }
    }

    fn parse_parenthesized_expression(&mut self) {
        let (_, left) = self.expect(Token::OpeningParenthesis);
        self.parse_expression();
        let (_, right) = self.expect(Token::ClosingParenthesis);
        self.actions
            .apply_operator((ExprOperator::Parentheses, left.extend(&right)))
    }

    fn parse_infix_expr(&mut self) {
        self.parse_atomic_expr();
        while self.lookahead() == Token::Plus {
            let (_, plus_span) = self.bump();
            self.parse_atomic_expr();
            self.actions.apply_operator((ExprOperator::Plus, plus_span));
        }
    }

    fn parse_atomic_expr(&mut self) {
        let (token, span) = self.bump();
        match token {
            Token::Ident(ident) => self.actions.push_atom((ExprAtom::Ident(ident), span)),
            Token::Literal(literal) => self.actions.push_atom((ExprAtom::Literal(literal), span)),
            _ => self.actions.emit_diagnostic(InternalDiagnostic::new(
                Message::UnexpectedToken,
                iter::empty(),
                span,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ast::*;
    use super::Token::*;
    use super::*;
    use diagnostics::{DiagnosticsListener, InternalDiagnostic, Message};
    use frontend::syntax::{self, ExprAtom, ExprOperator};
    use std::cell::RefCell;
    use std::collections::HashMap;

    #[test]
    fn parse_empty_src() {
        assert_eq_actions(input_tokens![], file([]))
    }

    struct TestContext {
        actions: RefCell<Vec<Action>>,
        token_seq_kind: Option<TokenSeqKind>,
    }

    impl TestContext {
        fn new() -> TestContext {
            TestContext {
                actions: RefCell::new(Vec::new()),
                token_seq_kind: None,
            }
        }
    }

    enum TokenSeqKind {
        MacroArg,
        MacroDef,
    }

    impl<'a> DiagnosticsListener<SymRange<usize>> for &'a mut TestContext {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymRange<usize>>) {
            self.actions.borrow_mut().push(Action::Error(diagnostic))
        }
    }

    impl<'a> syntax::FileContext<SymIdent, SymCommand, SymLiteral, SymRange<usize>>
        for &'a mut TestContext
    {
        type LineActions = Self;

        fn enter_line(self, label: Option<(SymIdent, SymRange<usize>)>) -> Self::LineActions {
            self.actions
                .borrow_mut()
                .push(Action::EnterLine(label.map(|(ident, _)| ident)));
            self
        }
    }

    impl<'a> syntax::LineActions<SymIdent, SymCommand, SymLiteral, SymRange<usize>>
        for &'a mut TestContext
    {
        type CommandContext = Self;
        type MacroParamsActions = Self;
        type MacroInvocationContext = Self;
        type Parent = Self;

        fn enter_command(
            self,
            (command, _): (SymCommand, SymRange<usize>),
        ) -> Self::CommandContext {
            self.actions
                .borrow_mut()
                .push(Action::EnterInstruction(command));
            self
        }

        fn enter_macro_def(self) -> Self::MacroParamsActions {
            self.actions.borrow_mut().push(Action::EnterMacroDef);
            self.token_seq_kind = Some(TokenSeqKind::MacroDef);
            self
        }

        fn enter_macro_invocation(
            self,
            name: (SymIdent, SymRange<usize>),
        ) -> Self::MacroInvocationContext {
            self.actions
                .borrow_mut()
                .push(Action::EnterMacroInvocation(name.0));
            self
        }

        fn exit(self) -> Self::Parent {
            self.actions.borrow_mut().push(Action::ExitLine);
            self
        }
    }

    impl<'a> syntax::CommandContext<SymRange<usize>> for &'a mut TestContext {
        type Command = SymCommand;
        type Ident = SymIdent;
        type Literal = SymLiteral;
        type ArgActions = ArgContext<'a>;
        type Parent = Self;

        fn add_argument(self) -> Self::ArgActions {
            self.actions.borrow_mut().push(Action::EnterArgument);
            ArgContext::new(self)
        }

        fn exit(self) -> Self::Parent {
            self.actions.borrow_mut().push(Action::ExitInstruction);
            self
        }
    }

    struct ArgContext<'a> {
        expr_context: ExprContext,
        parent: &'a mut TestContext,
    }

    impl<'a> ArgContext<'a> {
        fn new(parent: &'a mut TestContext) -> Self {
            ArgContext {
                expr_context: ExprContext::new(),
                parent,
            }
        }
    }

    impl<'a> DiagnosticsListener<SymRange<usize>> for ArgContext<'a> {
        fn emit_diagnostic(&mut self, _: InternalDiagnostic<SymRange<usize>>) {
            unimplemented!()
        }
    }

    impl<'a> syntax::ExprActions<SymRange<usize>> for ArgContext<'a> {
        type Ident = SymIdent;
        type Literal = SymLiteral;
        type Parent = &'a mut TestContext;

        fn push_atom(&mut self, atom: (ExprAtom<Self::Ident, Self::Literal>, SymRange<usize>)) {
            self.expr_context.push_atom(atom)
        }

        fn apply_operator(&mut self, operator: (ExprOperator, SymRange<usize>)) {
            self.expr_context.apply_operator(operator)
        }

        fn exit(self) -> Self::Parent {
            {
                let mut actions = self.parent.actions.borrow_mut();
                actions.push(Action::AcceptExpr(self.expr_context.rpn_expr));
                actions.push(Action::ExitArgument)
            }
            self.parent
        }
    }

    struct ExprContext {
        rpn_expr: TestRpnExpr,
        diagnostics: Vec<InternalDiagnostic<SymRange<usize>>>,
    }

    type TestRpnExpr = RpnExpr<SymIdent, SymLiteral, SymRange<usize>>;

    impl ExprContext {
        fn new() -> Self {
            ExprContext {
                rpn_expr: Vec::new(),
                diagnostics: Vec::new(),
            }
        }
    }

    impl DiagnosticsListener<SymRange<usize>> for ExprContext {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymRange<usize>>) {
            self.diagnostics.push(diagnostic)
        }
    }

    impl syntax::ExprActions<SymRange<usize>> for ExprContext {
        type Ident = SymIdent;
        type Literal = SymLiteral;
        type Parent = (TestRpnExpr, Vec<InternalDiagnostic<SymRange<usize>>>);

        fn push_atom(&mut self, atom: (ExprAtom<Self::Ident, Self::Literal>, SymRange<usize>)) {
            self.rpn_expr.push((RpnAction::Push(atom.0), atom.1))
        }

        fn apply_operator(&mut self, operator: (ExprOperator, SymRange<usize>)) {
            self.rpn_expr
                .push((RpnAction::Apply(operator.0), operator.1))
        }

        fn exit(self) -> Self::Parent {
            (self.rpn_expr, self.diagnostics)
        }
    }

    impl<'a> syntax::MacroParamsActions<SymRange<usize>> for &'a mut TestContext {
        type Ident = SymIdent;
        type Command = SymCommand;
        type Literal = SymLiteral;
        type MacroBodyActions = Self;
        type Parent = Self;

        fn add_parameter(&mut self, (ident, _): (SymIdent, SymRange<usize>)) {
            self.actions.borrow_mut().push(Action::AddParameter(ident))
        }

        fn exit(self) -> Self::MacroBodyActions {
            self.actions.borrow_mut().push(Action::EnterMacroBody);
            self
        }
    }

    impl<'a> syntax::MacroInvocationContext<SymRange<usize>> for &'a mut TestContext {
        type Token = SymToken;
        type Parent = Self;
        type MacroArgContext = Self;

        fn enter_macro_arg(self) -> Self::MacroArgContext {
            self.actions.borrow_mut().push(Action::EnterMacroArg);
            self.token_seq_kind = Some(TokenSeqKind::MacroArg);
            self
        }

        fn exit(self) -> Self::Parent {
            self.actions.borrow_mut().push(Action::ExitMacroInvocation);
            self
        }
    }

    impl<'a> syntax::TokenSeqContext<SymRange<usize>> for &'a mut TestContext {
        type Token = SymToken;
        type Parent = Self;

        fn push_token(&mut self, token: (Self::Token, SymRange<usize>)) {
            let id = token.1.start;
            assert_eq!(id, token.1.end);
            self.actions.borrow_mut().push(Action::PushTerminal(id))
        }

        fn exit(self) -> Self::Parent {
            self.actions
                .borrow_mut()
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
        assert_eq_actions(
            input_tokens![Eol],
            file([unlabeled(empty()), unlabeled(empty())]),
        )
    }

    fn assert_eq_actions(mut input: InputTokens, expected: File) {
        if input
            .tokens
            .last()
            .map(|token| token.kind() != Token::Eof)
            .unwrap_or(true)
        {
            input.tokens.push(Token::Eof)
        }
        let mut parsing_context = TestContext::new();
        parse_src(with_spans(&input.tokens), &mut parsing_context);
        assert_eq!(
            parsing_context.actions.into_inner(),
            expected.into_actions(&input)
        )
    }

    fn with_spans<'a>(
        tokens: impl IntoIterator<Item = &'a SymToken>,
    ) -> impl Iterator<Item = (SymToken, SymRange<usize>)> {
        tokens
            .into_iter()
            .cloned()
            .zip((0..).map(|n| SymRange::from(n)))
    }

    #[test]
    fn parse_nullary_instruction() {
        assert_eq_actions(
            input_tokens![nop @ Command(())],
            file([unlabeled(command("nop", []))]),
        )
    }

    #[test]
    fn parse_nullary_instruction_after_eol() {
        assert_eq_actions(
            input_tokens![Eol, nop @ Command(())],
            file([unlabeled(empty()), unlabeled(command("nop", []))]),
        )
    }

    #[test]
    fn parse_nullary_instruction_followed_by_eol() {
        assert_eq_actions(
            input_tokens![daa @ Command(()), Eol],
            file([unlabeled(command("daa", [])), unlabeled(empty())]),
        )
    }

    #[test]
    fn parse_unary_instruction() {
        assert_eq_actions(
            input_tokens![db @ Command(()), my_ptr @ Ident(())],
            file([unlabeled(command("db", [expr().ident("my_ptr")]))]),
        )
    }

    #[test]
    fn parse_binary_instruction() {
        assert_eq_actions(
            input_tokens![Command(()), Ident(()), Comma, Literal(())],
            file([unlabeled(command(0, [expr().ident(1), expr().literal(3)]))]),
        );
    }

    #[test]
    fn parse_two_instructions() {
        let tokens = input_tokens![
            Command(()),
            Ident(()),
            Comma,
            Literal(()),
            Eol,
            ld @ Command(()),
            a @ Literal(()),
            Comma,
            some_const @ Ident(()),
        ];
        let expected = file([
            unlabeled(command(0, [expr().ident(1), expr().literal(3)])),
            unlabeled(command(
                "ld",
                [expr().literal("a"), expr().ident("some_const")],
            )),
        ]);
        assert_eq_actions(tokens, expected)
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
        let expected = file([
            unlabeled(command(0, [expr().literal(1), expr().ident(3)])),
            unlabeled(empty()),
            unlabeled(command(6, [expr().ident(7), expr().literal(9)])),
        ]);
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_empty_macro_definition() {
        let tokens = input_tokens![Ident(()), Colon, Macro, Eol, Endm];
        let expected_actions = file([labeled(0, macro_def([], [], 4))]);
        assert_eq_actions(tokens, expected_actions);
    }

    #[test]
    fn parse_macro_definition_with_instruction() {
        let tokens = input_tokens![Ident(()), Colon, Macro, Eol, Command(()), Eol, Endm];
        let expected_actions = file([labeled(0, macro_def([], [4, 5], 6))]);
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
        let expected = file([labeled(0, macro_def([3, 5], [7, 8], 9))]);
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_label() {
        let tokens = input_tokens![Ident(()), Colon, Eol];
        let expected_actions = file([labeled(0, empty()), unlabeled(empty())]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_labeled_instruction() {
        let tokens = input_tokens![Ident(()), Colon, Command(()), Eol];
        let expected = file([labeled(0, command(2, [])), unlabeled(empty())]);
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_deref_operand() {
        let tokens = input_tokens![
            jp @ Command(()),
            open @ OpeningParenthesis,
            hl @ Literal(()),
            close @ ClosingParenthesis,
        ];
        let expected = file([unlabeled(command(
            "jp",
            [expr().literal("hl").parentheses("open", "close")],
        ))]);
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_nullary_macro_invocation() {
        let tokens = input_tokens![Ident(())];
        let expected_actions = file([unlabeled(invoke(0, []))]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_invocation() {
        let tokens = input_tokens![Ident(()), Literal(())];
        let expected_actions = file([unlabeled(invoke(0, [token_seq([1])]))]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_invocation_with_multiple_terminals() {
        let tokens = input_tokens![Ident(()), Literal(()), Literal(()), Literal(())];
        let expected_actions = file([unlabeled(invoke(0, [token_seq([1, 2, 3])]))]);
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
        let expected_actions = file([unlabeled(invoke(
            0,
            [token_seq([1, 2]), token_seq([4, 5, 6])],
        ))]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_sum_arg() {
        let tokens = input_tokens![
            Command(()),
            x @ Ident(()),
            plus @ Plus,
            y @ Literal(()),
        ];
        let expected_actions = file([unlabeled(command(
            0,
            [expr().ident("x").literal("y").plus("plus")],
        ))]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_long_sum_arg() {
        let tokens = input_tokens![
            x @ Ident(()),
            plus1 @ Plus,
            y @ Literal(()),
            plus2 @ Plus,
            z @ Ident(()),
        ];
        let expected = expr()
            .ident("x")
            .literal("y")
            .plus("plus1")
            .ident("z")
            .plus("plus2");
        assert_eq_rpn_expr(tokens, expected)
    }

    fn assert_eq_rpn_expr(mut input: InputTokens, expected: SymExpr) {
        let (parsed_rpn_expr, _) = parse_sym_expr(&mut input);
        assert_eq!(parsed_rpn_expr, expected.resolve(&input));
    }

    fn assert_eq_expr_diagnostics(
        mut input: InputTokens,
        expected: InternalDiagnostic<SymRange<TokenRef>>,
    ) {
        let (_, diagnostics) = parse_sym_expr(&mut input);
        assert_eq!(diagnostics, [SymDiagnostic(expected).resolve(&input)])
    }

    fn parse_sym_expr(
        input: &mut InputTokens,
    ) -> (TestRpnExpr, Vec<InternalDiagnostic<SymRange<usize>>>) {
        input.tokens.push(Token::Eof);
        let tokens = &mut with_spans(&input.tokens).peekable();
        let parser = ExprParser {
            tokens,
            actions: ExprContext::new(),
        };
        parser.parse()
    }

    #[test]
    fn diagnose_stmt_starting_with_literal() {
        assert_eq_actions(
            input_tokens![a @ Literal(())],
            file([unlabeled(line_error(Message::UnexpectedToken, ["a"], "a"))]),
        )
    }

    #[test]
    fn diagnose_missing_comma_in_arg_list() {
        assert_eq_actions(
            input_tokens![Command(()), Literal(()), unexpected @ Literal(())],
            file([unlabeled(malformed_command(
                0,
                [expr().literal(1)],
                arg_error(Message::UnexpectedToken, ["unexpected"], "unexpected"),
            ))]),
        )
    }

    #[test]
    fn diagnose_eof_after_macro_param_list() {
        assert_eq_actions(
            input_tokens![Macro, eof @ Eof],
            file([unlabeled(malformed_macro_def_head(
                [],
                arg_error(Message::UnexpectedEof, [], "eof"),
            ))]),
        )
    }

    #[test]
    fn diagnose_eof_in_macro_body() {
        assert_eq_actions(
            input_tokens![Macro, Eol, eof @ Eof],
            file([unlabeled(malformed_macro_def(
                [],
                [],
                arg_error(Message::UnexpectedEof, [], "eof"),
            ))]),
        )
    }

    #[test]
    fn diagnose_unexpected_token_in_expr() {
        let input = input_tokens![plus @ Plus];
        assert_eq_expr_diagnostics(
            input,
            InternalDiagnostic::new(
                Message::UnexpectedToken,
                iter::empty(),
                TokenRef::from("plus").into(),
            ),
        )
    }
}
