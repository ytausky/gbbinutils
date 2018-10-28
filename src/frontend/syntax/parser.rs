use super::*;
use diagnostics::{InternalDiagnostic, Message};
use span::Span;

use std::iter;

type TokenKind = Token<(), (), (), ()>;

impl Copy for TokenKind {}

impl<C, I, L, E> Token<C, I, L, E> {
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
            Error(_) => Error(()),
            Ident(_) => Ident(()),
            Literal(_) => Literal(()),
            Macro => Macro,
            OpeningParenthesis => OpeningParenthesis,
            Plus => Plus,
        }
    }
}

const LINE_FOLLOW_SET: &[TokenKind] = &[Token::Eol, Token::Eof];

pub fn parse_src<Id, C, L, S, I, F>(tokens: I, context: F) -> F
where
    S: Span,
    I: Iterator<Item = (Token<Id, C, L>, S)>,
    F: FileContext<Id, C, L, S>,
{
    let mut parser = Parser {
        tokens: tokens.peekable(),
    };
    parser.parse_file(context)
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

    fn parse_file<F: FileContext<Id, C, L, S>>(&mut self, context: F) -> F {
        self.parse_terminated_list(Token::Eol, &[Token::Eof], |p, c| p.parse_stmt(c), context)
    }

    fn parse_stmt<F: FileContext<Id, C, L, S>>(&mut self, context: F) -> F {
        if self.lookahead() == Token::Ident(()) {
            self.parse_potentially_labeled_stmt(context)
        } else {
            self.parse_unlabeled_stmt(context.enter_stmt(None)).exit()
        }
    }

    fn parse_potentially_labeled_stmt<F: FileContext<Id, C, L, S>>(&mut self, context: F) -> F {
        let ident = self.expect_ident();
        if self.consume(Token::Colon) {
            self.parse_unlabeled_stmt(context.enter_stmt(Some(ident)))
        } else {
            self.parse_macro_invocation(ident, context.enter_stmt(None))
        }.exit()
    }

    fn parse_unlabeled_stmt<LA: StmtContext<Id, C, L, S>>(&mut self, mut context: LA) -> LA {
        match self.lookahead() {
            Token::Eol | Token::Eof => context,
            Token::Command(()) => self.parse_command(context),
            Token::Ident(()) => {
                let ident = self.expect_ident();
                self.parse_macro_invocation(ident, context)
            }
            Token::Macro => self.parse_macro_def(context),
            _ => {
                let (_, range) = self.bump();
                context.emit_diagnostic(InternalDiagnostic::new(
                    Message::UnexpectedToken,
                    vec![range.clone()],
                    range,
                ));
                context
            }
        }
    }

    fn parse_macro_def<LA: StmtContext<Id, C, L, S>>(&mut self, context: LA) -> LA {
        let (_, span) = self.expect(Token::Macro);
        let mut context = self.parse_terminated_list(
            Token::Comma,
            LINE_FOLLOW_SET,
            |p, a| p.parse_macro_param(a),
            context.enter_macro_def(span),
        );
        if self.consume(Token::Eol) {
            let mut body_context = context.exit();
            self.take_token_while(
                |x| x != Token::Endm && x != Token::Eof,
                |token| body_context.push_token(token),
            );
            if self.lookahead() == Token::Endm {
                let endm = self.bump();
                body_context.push_token((Token::Eof, endm.1));
            } else {
                assert_eq!(self.lookahead(), Token::Eof);
                body_context.emit_diagnostic(InternalDiagnostic::new(
                    Message::UnexpectedEof,
                    iter::empty(),
                    self.tokens.peek().unwrap().1.clone(),
                ))
            }
            body_context
        } else {
            assert_eq!(self.lookahead(), Token::Eof);
            context.emit_diagnostic(InternalDiagnostic::new(
                Message::UnexpectedEof,
                iter::empty(),
                self.tokens.peek().unwrap().1.clone(),
            ));
            context.exit()
        }.exit()
    }

    fn parse_macro_param<MPA>(&mut self, mut context: MPA) -> MPA
    where
        MPA: MacroParamsContext<S, Command = C, Ident = Id, Literal = L>,
    {
        context.add_parameter(self.expect_ident());
        context
    }

    fn parse_command<LA: StmtContext<Id, C, L, S>>(&mut self, context: LA) -> LA {
        let first_token = self.expect_command();
        let mut command_context = context.enter_command(first_token);
        command_context = self.parse_argument_list(command_context);
        command_context.exit()
    }

    fn parse_macro_invocation<LA: StmtContext<Id, C, L, S>>(
        &mut self,
        name: (Id, S),
        context: LA,
    ) -> LA {
        let mut invocation_context = context.enter_macro_invocation(name);
        invocation_context = self.parse_macro_arg_list(invocation_context);
        invocation_context.exit()
    }

    fn parse_argument_list<CC: CommandContext<S, Command = C, Ident = Id, Literal = L>>(
        &mut self,
        context: CC,
    ) -> CC {
        self.parse_terminated_list(
            Token::Comma,
            LINE_FOLLOW_SET,
            |p, c| p.parse_argument(c),
            context,
        )
    }

    fn parse_macro_arg_list<M: MacroInvocationContext<S, Token = Token<Id, C, L>>>(
        &mut self,
        context: M,
    ) -> M {
        self.parse_terminated_list(
            Token::Comma,
            LINE_FOLLOW_SET,
            |p, context| {
                let mut arg_context = context.enter_macro_arg();
                p.take_token_while(
                    |x| x != Token::Comma && x != Token::Eol && x != Token::Eof,
                    |token| arg_context.push_token(token),
                );
                arg_context.exit()
            },
            context,
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
        mut context: A,
    ) -> A
    where
        P: FnMut(&mut Self, A) -> A,
        A: DiagnosticsListener<S>,
    {
        context = parser(self, context);
        while self.consume(delimiter) {
            context = parser(self, context)
        }
        context
    }

    fn parse_argument<CC: CommandContext<S, Command = C, Ident = Id, Literal = L>>(
        &mut self,
        context: CC,
    ) -> CC {
        ExprParser {
            tokens: self,
            context: context.add_argument(),
        }.parse()
    }
}

struct ExprParser<'a, T: 'a, A> {
    tokens: &'a mut T,
    context: A,
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
    A: ExprContext<T::Span, Ident = T::Ident, Literal = T::Literal>,
{
    fn parse(mut self) -> A::Parent {
        self.parse_expression();
        self.context.exit()
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
        if self.lookahead() == Token::ClosingParenthesis {
            let (_, right) = self.expect(Token::ClosingParenthesis);
            self.context
                .apply_operator((ExprOperator::Parentheses, left.extend(&right)))
        } else {
            self.context.emit_diagnostic(InternalDiagnostic::new(
                Message::UnmatchedParenthesis,
                iter::empty(),
                left,
            ));
        }
    }

    fn parse_infix_expr(&mut self) {
        self.parse_atomic_expr();
        while self.lookahead() == Token::Plus {
            let (_, plus_span) = self.bump();
            self.parse_atomic_expr();
            self.context.apply_operator((ExprOperator::Plus, plus_span));
        }
    }

    fn parse_atomic_expr(&mut self) {
        if self.lookahead() == Token::Eof {
            return;
        }

        let (token, span) = self.bump();
        match token {
            Token::Ident(ident) => self.context.push_atom((ExprAtom::Ident(ident), span)),
            Token::Literal(literal) => self.context.push_atom((ExprAtom::Literal(literal), span)),
            _ => self.context.emit_diagnostic(InternalDiagnostic::new(
                Message::UnexpectedToken,
                iter::once(span.clone()),
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
    use frontend::syntax::{ExprAtom, ExprOperator};
    use std::borrow::Borrow;
    use std::collections::HashMap;

    #[test]
    fn parse_empty_src() {
        assert_eq_actions(input_tokens![], [])
    }

    struct FileActionCollector {
        actions: Vec<FileAction<SymRange<TokenRef>>>,
    }

    impl FileActionCollector {
        fn new() -> FileActionCollector {
            FileActionCollector {
                actions: Vec::new(),
            }
        }
    }

    impl DiagnosticsListener<SymRange<TokenRef>> for FileActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymRange<TokenRef>>) {
            self.actions.push(FileAction::EmitDiagnostic(diagnostic))
        }
    }

    impl FileContext<SymIdent, SymCommand, SymLiteral, SymRange<TokenRef>> for FileActionCollector {
        type StmtContext = StmtActionCollector;

        fn enter_stmt(self, label: Option<(SymIdent, SymRange<TokenRef>)>) -> StmtActionCollector {
            StmtActionCollector {
                label,
                actions: Vec::new(),
                parent: self,
            }
        }
    }

    struct StmtActionCollector {
        label: Option<(SymIdent, SymRange<TokenRef>)>,
        actions: Vec<StmtAction<SymRange<TokenRef>>>,
        parent: FileActionCollector,
    }

    impl DiagnosticsListener<SymRange<TokenRef>> for StmtActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymRange<TokenRef>>) {
            self.actions.push(StmtAction::EmitDiagnostic(diagnostic))
        }
    }

    impl StmtContext<SymIdent, SymCommand, SymLiteral, SymRange<TokenRef>> for StmtActionCollector {
        type CommandContext = CommandActionCollector;
        type MacroParamsContext = MacroParamsActionCollector;
        type MacroInvocationContext = MacroInvocationActionCollector;
        type Parent = FileActionCollector;

        fn enter_command(
            self,
            command: (SymCommand, SymRange<TokenRef>),
        ) -> CommandActionCollector {
            CommandActionCollector {
                command,
                actions: Vec::new(),
                parent: self,
            }
        }

        fn enter_macro_def(self, keyword: SymRange<TokenRef>) -> MacroParamsActionCollector {
            MacroParamsActionCollector {
                keyword,
                actions: Vec::new(),
                parent: self,
            }
        }

        fn enter_macro_invocation(
            self,
            name: (SymIdent, SymRange<TokenRef>),
        ) -> MacroInvocationActionCollector {
            MacroInvocationActionCollector {
                name,
                actions: Vec::new(),
                parent: self,
            }
        }

        fn exit(mut self) -> FileActionCollector {
            self.parent.actions.push(FileAction::Stmt {
                label: self.label,
                actions: self.actions,
            });
            self.parent
        }
    }

    struct CommandActionCollector {
        command: (SymCommand, SymRange<TokenRef>),
        actions: Vec<CommandAction<SymRange<TokenRef>>>,
        parent: StmtActionCollector,
    }

    impl DiagnosticsListener<SymRange<TokenRef>> for CommandActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymRange<TokenRef>>) {
            self.actions.push(CommandAction::EmitDiagnostic(diagnostic))
        }
    }

    impl CommandContext<SymRange<TokenRef>> for CommandActionCollector {
        type Command = SymCommand;
        type Ident = SymIdent;
        type Literal = SymLiteral;
        type ArgContext = ArgActionCollector;
        type Parent = StmtActionCollector;

        fn add_argument(self) -> ArgActionCollector {
            ArgActionCollector {
                expr_action_collector: ExprActionCollector::new(),
                parent: self,
            }
        }

        fn exit(mut self) -> StmtActionCollector {
            self.parent.actions.push(StmtAction::Command {
                command: self.command,
                actions: self.actions,
            });
            self.parent
        }
    }

    struct ArgActionCollector {
        expr_action_collector: ExprActionCollector,
        parent: CommandActionCollector,
    }

    impl DiagnosticsListener<SymRange<TokenRef>> for ArgActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymRange<TokenRef>>) {
            self.expr_action_collector.emit_diagnostic(diagnostic)
        }
    }

    impl ExprContext<SymRange<TokenRef>> for ArgActionCollector {
        type Ident = SymIdent;
        type Literal = SymLiteral;
        type Parent = CommandActionCollector;

        fn push_atom(&mut self, atom: (ExprAtom<SymIdent, SymLiteral>, SymRange<TokenRef>)) {
            self.expr_action_collector.push_atom(atom)
        }

        fn apply_operator(&mut self, operator: (ExprOperator, SymRange<TokenRef>)) {
            self.expr_action_collector.apply_operator(operator)
        }

        fn exit(mut self) -> CommandActionCollector {
            self.parent.actions.push(CommandAction::AddArgument {
                actions: self.expr_action_collector.exit(),
            });
            self.parent
        }
    }

    struct ExprActionCollector {
        actions: Vec<ExprAction<SymRange<TokenRef>>>,
    }

    impl ExprActionCollector {
        fn new() -> ExprActionCollector {
            ExprActionCollector {
                actions: Vec::new(),
            }
        }
    }

    impl DiagnosticsListener<SymRange<TokenRef>> for ExprActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymRange<TokenRef>>) {
            self.actions.push(ExprAction::EmitDiagnostic(diagnostic))
        }
    }

    impl ExprContext<SymRange<TokenRef>> for ExprActionCollector {
        type Ident = SymIdent;
        type Literal = SymLiteral;
        type Parent = Vec<ExprAction<SymRange<TokenRef>>>;

        fn push_atom(&mut self, atom: (ExprAtom<SymIdent, SymLiteral>, SymRange<TokenRef>)) {
            self.actions.push(ExprAction::PushAtom(atom))
        }

        fn apply_operator(&mut self, operator: (ExprOperator, SymRange<TokenRef>)) {
            self.actions.push(ExprAction::ApplyOperator(operator))
        }

        fn exit(self) -> Self::Parent {
            self.actions
        }
    }

    struct MacroParamsActionCollector {
        keyword: SymRange<TokenRef>,
        actions: Vec<MacroParamsAction<SymRange<TokenRef>>>,
        parent: StmtActionCollector,
    }

    impl DiagnosticsListener<SymRange<TokenRef>> for MacroParamsActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymRange<TokenRef>>) {
            self.actions
                .push(MacroParamsAction::EmitDiagnostic(diagnostic))
        }
    }

    impl MacroParamsContext<SymRange<TokenRef>> for MacroParamsActionCollector {
        type Command = SymCommand;
        type Ident = SymIdent;
        type Literal = SymLiteral;
        type MacroBodyContext = MacroBodyActionCollector;
        type Parent = StmtActionCollector;

        fn add_parameter(&mut self, param: (SymIdent, SymRange<TokenRef>)) {
            self.actions.push(MacroParamsAction::AddParameter(param))
        }

        fn exit(self) -> MacroBodyActionCollector {
            MacroBodyActionCollector {
                actions: Vec::new(),
                parent: self,
            }
        }
    }

    struct MacroBodyActionCollector {
        actions: Vec<TokenSeqAction<SymRange<TokenRef>>>,
        parent: MacroParamsActionCollector,
    }

    impl DiagnosticsListener<SymRange<TokenRef>> for MacroBodyActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymRange<TokenRef>>) {
            self.actions
                .push(TokenSeqAction::EmitDiagnostic(diagnostic))
        }
    }

    impl TokenSeqContext<SymRange<TokenRef>> for MacroBodyActionCollector {
        type Token = Token<SymIdent, SymCommand, SymLiteral>;
        type Parent = StmtActionCollector;

        fn push_token(&mut self, token: (Self::Token, SymRange<TokenRef>)) {
            self.actions.push(TokenSeqAction::PushToken(token))
        }

        fn exit(mut self) -> StmtActionCollector {
            self.parent.parent.actions.push(StmtAction::MacroDef {
                keyword: self.parent.keyword,
                params: self.parent.actions,
                body: self.actions,
            });
            self.parent.parent
        }
    }

    struct MacroInvocationActionCollector {
        name: (SymIdent, SymRange<TokenRef>),
        actions: Vec<MacroInvocationAction<SymRange<TokenRef>>>,
        parent: StmtActionCollector,
    }

    impl DiagnosticsListener<SymRange<TokenRef>> for MacroInvocationActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymRange<TokenRef>>) {
            self.actions
                .push(MacroInvocationAction::EmitDiagnostic(diagnostic))
        }
    }

    impl MacroInvocationContext<SymRange<TokenRef>> for MacroInvocationActionCollector {
        type Token = Token<SymIdent, SymCommand, SymLiteral>;
        type MacroArgContext = MacroArgActionCollector;
        type Parent = StmtActionCollector;

        fn enter_macro_arg(self) -> MacroArgActionCollector {
            MacroArgActionCollector {
                actions: Vec::new(),
                parent: self,
            }
        }

        fn exit(mut self) -> StmtActionCollector {
            self.parent.actions.push(StmtAction::MacroInvocation {
                name: self.name,
                actions: self.actions,
            });
            self.parent
        }
    }

    struct MacroArgActionCollector {
        actions: Vec<TokenSeqAction<SymRange<TokenRef>>>,
        parent: MacroInvocationActionCollector,
    }

    impl DiagnosticsListener<SymRange<TokenRef>> for MacroArgActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymRange<TokenRef>>) {
            self.actions
                .push(TokenSeqAction::EmitDiagnostic(diagnostic))
        }
    }

    impl TokenSeqContext<SymRange<TokenRef>> for MacroArgActionCollector {
        type Token = Token<SymIdent, SymCommand, SymLiteral>;
        type Parent = MacroInvocationActionCollector;

        fn push_token(&mut self, token: (Self::Token, SymRange<TokenRef>)) {
            self.actions.push(TokenSeqAction::PushToken(token))
        }

        fn exit(mut self) -> MacroInvocationActionCollector {
            self.parent
                .actions
                .push(MacroInvocationAction::MacroArg(self.actions));
            self.parent
        }
    }

    #[test]
    fn parse_empty_stmt() {
        assert_eq_actions(input_tokens![Eol], [unlabeled(empty()), unlabeled(empty())])
    }

    fn assert_eq_actions(
        input: InputTokens,
        expected: impl Borrow<[FileAction<SymRange<TokenRef>>]>,
    ) {
        let mut parsing_context = FileActionCollector::new();
        parsing_context = parse_src(with_spans(&input.tokens), parsing_context);
        assert_eq!(parsing_context.actions, expected.borrow())
    }

    fn with_spans<'a>(
        tokens: impl IntoIterator<Item = &'a (SymToken, TokenRef)>,
    ) -> impl Iterator<Item = (SymToken, SymRange<TokenRef>)> {
        tokens.into_iter().cloned().map(|(t, r)| (t, r.into()))
    }

    #[test]
    fn parse_nullary_instruction() {
        assert_eq_actions(
            input_tokens![nop @ Command(())],
            [unlabeled(command("nop", []))],
        )
    }

    #[test]
    fn parse_nullary_instruction_after_eol() {
        assert_eq_actions(
            input_tokens![Eol, nop @ Command(())],
            [unlabeled(empty()), unlabeled(command("nop", []))],
        )
    }

    #[test]
    fn parse_nullary_instruction_followed_by_eol() {
        assert_eq_actions(
            input_tokens![daa @ Command(()), Eol],
            [unlabeled(command("daa", [])), unlabeled(empty())],
        )
    }

    #[test]
    fn parse_unary_instruction() {
        assert_eq_actions(
            input_tokens![db @ Command(()), my_ptr @ Ident(())],
            [unlabeled(command("db", [expr().ident("my_ptr")]))],
        )
    }

    #[test]
    fn parse_binary_instruction() {
        assert_eq_actions(
            input_tokens![Command(()), Ident(()), Comma, Literal(())],
            [unlabeled(command(0, [expr().ident(1), expr().literal(3)]))],
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
        let expected = [
            unlabeled(command(0, [expr().ident(1), expr().literal(3)])),
            unlabeled(command(
                "ld",
                [expr().literal("a"), expr().ident("some_const")],
            )),
        ];
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
        let expected = [
            unlabeled(command(0, [expr().literal(1), expr().ident(3)])),
            unlabeled(empty()),
            unlabeled(command(6, [expr().ident(7), expr().literal(9)])),
        ];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_empty_macro_definition() {
        let tokens = input_tokens![Ident(()), Colon, Macro, Eol, Endm];
        let expected_actions = [labeled(0, macro_def(2, [], Vec::new(), 4))];
        assert_eq_actions(tokens, expected_actions);
    }

    #[test]
    fn parse_macro_definition_with_instruction() {
        let tokens = input_tokens![Ident(()), Colon, Macro, Eol, Command(()), Eol, Endm];
        let expected_actions = [labeled(0, macro_def(2, [], tokens.token_seq([4, 5]), 6))];
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
        let expected = [labeled(
            0,
            macro_def(2, [3.into(), 5.into()], tokens.token_seq([7, 8]), 9),
        )];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_label() {
        let tokens = input_tokens![Ident(()), Colon, Eol];
        let expected_actions = [labeled(0, empty()), unlabeled(empty())];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_labeled_instruction() {
        let tokens = input_tokens![Ident(()), Colon, Command(()), Eol];
        let expected = [labeled(0, command(2, [])), unlabeled(empty())];
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
        let expected = [unlabeled(command(
            "jp",
            [expr().literal("hl").parentheses("open", "close")],
        ))];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_nullary_macro_invocation() {
        let tokens = input_tokens![Ident(())];
        let expected_actions = [unlabeled(invoke(0, []))];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_invocation() {
        let tokens = input_tokens![Ident(()), Literal(())];
        let expected_actions = [unlabeled(invoke(0, [tokens.token_seq([1])]))];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_invocation_with_multiple_terminals() {
        let tokens = input_tokens![Ident(()), Literal(()), Literal(()), Literal(())];
        let expected_actions = [unlabeled(invoke(0, [tokens.token_seq([1, 2, 3])]))];
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
        let expected_actions = [unlabeled(invoke(
            0,
            [tokens.token_seq([1, 2]), tokens.token_seq([4, 5, 6])],
        ))];
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
        let expected_actions = [unlabeled(command(
            0,
            [expr().ident("x").literal("y").plus("plus")],
        ))];
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

    fn assert_eq_rpn_expr(mut input: InputTokens, SymExpr(expected): SymExpr) {
        let parsed_rpn_expr = parse_sym_expr(&mut input);
        assert_eq!(parsed_rpn_expr, expected);
    }

    fn assert_eq_expr_diagnostics(
        mut input: InputTokens,
        expected: InternalDiagnostic<SymRange<TokenRef>>,
    ) {
        let expr_actions = parse_sym_expr(&mut input);
        assert_eq!(expr_actions, [ExprAction::EmitDiagnostic(expected)])
    }

    fn parse_sym_expr(input: &mut InputTokens) -> Vec<ExprAction<SymRange<TokenRef>>> {
        let tokens = &mut with_spans(&input.tokens).peekable();
        let parser = ExprParser {
            tokens,
            context: ExprActionCollector::new(),
        };
        parser.parse()
    }

    #[test]
    fn diagnose_stmt_starting_with_literal() {
        assert_eq_actions(
            input_tokens![a @ Literal(())],
            [unlabeled(stmt_error(Message::UnexpectedToken, ["a"], "a"))],
        )
    }

    #[test]
    fn diagnose_missing_comma_in_arg_list() {
        assert_eq_actions(
            input_tokens![Command(()), Literal(()), unexpected @ Literal(())],
            [unlabeled(malformed_command(
                0,
                [expr().literal(1)],
                arg_error(Message::UnexpectedToken, ["unexpected"], "unexpected"),
            ))],
        )
    }

    #[test]
    fn diagnose_eof_after_macro_param_list() {
        assert_eq_actions(
            input_tokens![Macro, eof @ Eof],
            [unlabeled(malformed_macro_def_head(
                0,
                [],
                arg_error(Message::UnexpectedEof, [], "eof"),
            ))],
        )
    }

    #[test]
    fn diagnose_eof_in_macro_body() {
        assert_eq_actions(
            input_tokens![Macro, Eol, eof @ Eof],
            [unlabeled(malformed_macro_def(
                0,
                [],
                Vec::new(),
                arg_error(Message::UnexpectedEof, [], "eof"),
            ))],
        )
    }

    #[test]
    fn diagnose_unexpected_token_in_expr() {
        let input = input_tokens![plus @ Plus];
        let span: SymRange<_> = TokenRef::from("plus").into();
        assert_eq_expr_diagnostics(
            input,
            InternalDiagnostic::new(Message::UnexpectedToken, iter::once(span.clone()), span),
        )
    }

    #[test]
    fn diagnose_unmatched_parentheses() {
        assert_eq_actions(
            input_tokens![Command(()), paren @ OpeningParenthesis],
            [unlabeled(command(
                0,
                [expr().error(Message::UnmatchedParenthesis, TokenRef::from("paren"))],
            ))],
        )
    }
}
