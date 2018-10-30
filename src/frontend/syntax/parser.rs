use super::*;
use diagnostics::{InternalDiagnostic, Message};
use span::Span;

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

pub fn parse_src<Id, C, L, S, I, F>(mut tokens: I, context: F) -> F
where
    S: Span,
    I: Iterator<Item = (Token<Id, C, L>, S)>,
    F: FileContext<Id, C, L, S>,
{
    let State { token, context } = State {
        token: tokens.next().unwrap(),
        context,
    }.parse_file(&mut tokens);
    assert_eq!(token.0.kind(), Token::Eof);
    context
}

trait TokenStream {
    type Ident;
    type Command;
    type Literal;
    type Span: Span;
    type Iter: Iterator<Item = (Token<Self::Ident, Self::Command, Self::Literal>, Self::Span)>;

    fn tokens(&mut self) -> &mut Self::Iter;

    fn bump(&mut self) -> <Self::Iter as Iterator>::Item {
        self.tokens().next().unwrap()
    }
}

impl<Id, C, L, S: Span, I: Iterator<Item = (Token<Id, C, L>, S)>> TokenStream for I {
    type Ident = Id;
    type Command = C;
    type Literal = L;
    type Span = S;
    type Iter = I;

    fn tokens(&mut self) -> &mut Self::Iter {
        self
    }
}

struct State<T, C> {
    token: T,
    context: C,
}

impl<T, C> State<T, C> {
    fn change_context<D, F: FnOnce(C) -> D>(self, f: F) -> State<T, D> {
        State {
            token: self.token,
            context: f(self.context),
        }
    }
}

impl<I, C, L, S, A> State<(Token<I, C, L>, S), A> {
    fn token_is_in(&self, kinds: &[TokenKind]) -> bool {
        kinds.iter().any(|x| *x == self.token.0.kind())
    }
}

impl<Id, C, L, S: Span, Ctx: FileContext<Id, C, L, S>> State<(Token<Id, C, L>, S), Ctx> {
    fn parse_file<I: Iterator<Item = (Token<Id, C, L>, S)>>(self, tokens: &mut I) -> Self {
        self.parse_terminated_list(Token::Eol, &[Token::Eof], |p, s| p.parse_stmt(s), tokens)
    }

    fn parse_stmt<I: Iterator<Item = (Token<Id, C, L>, S)>>(mut self, tokens: &mut I) -> Self {
        match self.token {
            (Token::Ident(ident), span) => {
                self.token = tokens.next().unwrap();
                self.parse_potentially_labeled_stmt((ident, span), tokens)
            }
            _ => self
                .change_context(|c| c.enter_stmt(None))
                .parse_unlabeled_stmt(tokens)
                .change_context(|c| c.exit()),
        }
    }

    fn parse_potentially_labeled_stmt<I: Iterator<Item = (Token<Id, C, L>, S)>>(
        mut self,
        ident: (Id, S),
        tokens: &mut I,
    ) -> Self {
        if let (Token::Colon, _) = self.token {
            self.token = tokens.next().unwrap();
            self.change_context(|c| c.enter_stmt(Some(ident)))
                .parse_unlabeled_stmt(tokens)
        } else {
            self.change_context(|c| c.enter_stmt(None))
                .parse_macro_invocation(ident, tokens)
        }.change_context(|c| c.exit())
    }
}

impl<Id, C, L, S: Span, Ctx: StmtContext<Id, C, L, S>> State<(Token<Id, C, L>, S), Ctx> {
    fn parse_unlabeled_stmt<I: Iterator<Item = (Token<Id, C, L>, S)>>(
        mut self,
        tokens: &mut I,
    ) -> Self {
        match self.token {
            (Token::Eol, _) | (Token::Eof, _) => self,
            (Token::Command(command), span) => {
                self.token = tokens.next().unwrap();
                self.parse_command((command, span), tokens)
            }
            (Token::Ident(ident), span) => {
                self.token = tokens.next().unwrap();
                self.parse_macro_invocation((ident, span), tokens)
            }
            (Token::Macro, span) => {
                self.token = tokens.next().unwrap();
                self.parse_macro_def(span, tokens)
            }
            (_, span) => {
                self.token = tokens.next().unwrap();
                self.context.emit_diagnostic(InternalDiagnostic::new(
                    Message::UnexpectedToken {
                        token: span.clone(),
                    },
                    span,
                ));
                self
            }
        }
    }

    fn parse_command<I: Iterator<Item = (Token<Id, C, L>, S)>>(
        self,
        command: (C, S),
        tokens: &mut I,
    ) -> Self {
        self.change_context(|c| c.enter_command(command))
            .parse_argument_list(tokens)
            .change_context(|c| c.exit())
    }

    fn parse_macro_def<I: Iterator<Item = (Token<Id, C, L>, S)>>(
        self,
        span: S,
        tokens: &mut I,
    ) -> Self {
        let mut state = self
            .change_context(|c| c.enter_macro_def(span))
            .parse_terminated_list(
                Token::Comma,
                LINE_FOLLOW_SET,
                |p, s| p.parse_macro_param(s),
                tokens,
            );
        if state.token.0.kind() == Token::Eol {
            state.token = tokens.next().unwrap();
            let mut state = state.change_context(|c| c.exit());
            loop {
                match state.token {
                    (Token::Endm, _) => {
                        state
                            .context
                            .push_token((Token::Eof, state.token.1.clone()));
                        state.token = tokens.next().unwrap();
                        break;
                    }
                    (Token::Eof, _) => {
                        state.context.emit_diagnostic(InternalDiagnostic::new(
                            Message::UnexpectedEof,
                            state.token.1.clone(),
                        ));
                        break;
                    }
                    other => {
                        state.context.push_token(other);
                        state.token = tokens.next().unwrap();
                    }
                }
            }
            state
        } else {
            assert_eq!(state.token.0.kind(), Token::Eof);
            state.context.emit_diagnostic(InternalDiagnostic::new(
                Message::UnexpectedEof,
                state.token.1.clone(),
            ));
            state.change_context(|c| c.exit())
        }.change_context(|c| c.exit())
    }

    fn parse_macro_invocation<I: Iterator<Item = (Token<Id, C, L>, S)>>(
        self,
        name: (Id, S),
        tokens: &mut I,
    ) -> Self {
        self.change_context(|c| c.enter_macro_invocation(name))
            .parse_macro_arg_list(tokens)
            .change_context(|c| c.exit())
    }
}

impl<Id, C, L, S: Span, Ctx: CommandContext<S, Command = C, Ident = Id, Literal = L>>
    State<(Token<Id, C, L>, S), Ctx>
{
    fn parse_argument_list<I: Iterator<Item = (Token<Id, C, L>, S)>>(self, tokens: &mut I) -> Self {
        self.parse_terminated_list(
            Token::Comma,
            LINE_FOLLOW_SET,
            |p, s| p.parse_argument(s),
            tokens,
        )
    }

    fn parse_argument<I: Iterator<Item = (Token<Id, C, L>, S)>>(self, tokens: &mut I) -> Self {
        ExprParser {
            tokens,
            context: self.context.add_argument(),
        }.parse(self.token)
    }
}

impl<Id, C, L, S: Span, Ctx: MacroParamsContext<S, Command = C, Ident = Id, Literal = L>>
    State<(Token<Id, C, L>, S), Ctx>
{
    fn parse_macro_param<I: Iterator<Item = (Token<Id, C, L>, S)>>(
        mut self,
        tokens: &mut I,
    ) -> Self {
        let ident = match self.token {
            (Token::Ident(ident), span) => (ident, span),
            _ => panic!(),
        };
        self.token = tokens.next().unwrap();
        self.context.add_parameter(ident);
        self
    }
}

impl<Id, C, L, S: Span, Ctx: MacroInvocationContext<S, Token = Token<Id, C, L>>>
    State<(Token<Id, C, L>, S), Ctx>
{
    fn parse_macro_arg_list<I: Iterator<Item = (Token<Id, C, L>, S)>>(
        self,
        tokens: &mut I,
    ) -> Self {
        self.parse_terminated_list(
            Token::Comma,
            LINE_FOLLOW_SET,
            |state, tokens| {
                let mut state = state.change_context(|c| c.enter_macro_arg());
                loop {
                    match state.token {
                        (Token::Comma, _) | (Token::Eol, _) | (Token::Eof, _) => break,
                        other => {
                            state.token = tokens.next().unwrap();
                            state.context.push_token(other)
                        }
                    }
                }
                state.change_context(|c| c.exit())
            },
            tokens,
        )
    }
}

impl<Id, C, L, S: Span, Ctx: DiagnosticsListener<S>> State<(Token<Id, C, L>, S), Ctx> {
    fn parse_terminated_list<P, I>(
        mut self,
        delimiter: TokenKind,
        terminators: &[TokenKind],
        parser: P,
        tokens: &mut I,
    ) -> Self
    where
        P: FnMut(Self, &mut I) -> Self,
        I: Iterator<Item = (Token<Id, C, L>, S)>,
    {
        self = self.parse_list(delimiter, terminators, parser, tokens);
        if !self.token_is_in(terminators) {
            let unexpected_span = self.token.1;
            self.context.emit_diagnostic(InternalDiagnostic::new(
                Message::UnexpectedToken {
                    token: unexpected_span.clone(),
                },
                unexpected_span,
            ));
            self.token = tokens.next().unwrap();
            while !self.token_is_in(terminators) {
                self.token = tokens.next().unwrap();
            }
        }
        self
    }

    fn parse_list<P, I>(
        self,
        delimiter: TokenKind,
        terminators: &[TokenKind],
        mut parser: P,
        tokens: &mut I,
    ) -> Self
    where
        P: FnMut(Self, &mut I) -> Self,
        I: Iterator<Item = (Token<Id, C, L>, S)>,
    {
        if self.token_is_in(terminators) {
            self
        } else {
            self.parse_nonempty_list(delimiter, &mut parser, tokens)
        }
    }

    fn parse_nonempty_list<P, I>(
        mut self,
        delimiter: TokenKind,
        parser: &mut P,
        tokens: &mut I,
    ) -> Self
    where
        P: FnMut(Self, &mut I) -> Self,
        I: Iterator<Item = (Token<Id, C, L>, S)>,
    {
        self = parser(self, tokens);
        while self.token.0.kind() == delimiter {
            self.token = tokens.next().unwrap();
            self = parser(self, tokens);
        }
        self
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

    fn tokens(&mut self) -> &mut Self::Iter {
        self.tokens.tokens()
    }
}

type ExprParsingResult<T> = Result<
    <<T as TokenStream>::Iter as Iterator>::Item,
    (
        <<T as TokenStream>::Iter as Iterator>::Item,
        InternalDiagnostic<<T as TokenStream>::Span>,
    ),
>;

impl<'a, T, A> ExprParser<'a, T, A>
where
    T: TokenStream + 'a,
    A: ExprContext<T::Span, Ident = T::Ident, Literal = T::Literal>,
{
    fn parse(
        mut self,
        mut token: <T::Iter as Iterator>::Item,
    ) -> State<<T::Iter as Iterator>::Item, A::Parent> {
        token = match self.parse_expression(token) {
            Ok(token) => token,
            Err((mut token, diagnostic)) => {
                self.context.emit_diagnostic(diagnostic);
                while !LINE_FOLLOW_SET.iter().any(|x| *x == token.0.kind()) {
                    token = self.bump();
                }
                token
            }
        };
        State {
            token,
            context: self.context.exit(),
        }
    }

    fn parse_expression(&mut self, mut token: <T::Iter as Iterator>::Item) -> ExprParsingResult<T> {
        match token {
            (Token::OpeningParenthesis, span) => {
                token = self.bump();
                self.parse_parenthesized_expression(span, token)
            }
            other => self.parse_infix_expr(other),
        }
    }

    fn parse_parenthesized_expression(
        &mut self,
        left: T::Span,
        token: <T::Iter as Iterator>::Item,
    ) -> ExprParsingResult<T> {
        let mut token = self.parse_expression(token)?;
        match token {
            (Token::ClosingParenthesis, right) => {
                token = self.bump();
                self.context
                    .apply_operator((ExprOperator::Parentheses, left.extend(&right)));
                Ok(token)
            }
            other => Err((
                other,
                InternalDiagnostic::new(Message::UnmatchedParenthesis, left),
            )),
        }
    }

    fn parse_infix_expr(&mut self, token: <T::Iter as Iterator>::Item) -> ExprParsingResult<T> {
        let mut token = self.parse_atomic_expr(token)?;
        while let (Token::Plus, span) = token {
            token = self.bump();
            token = self.parse_atomic_expr(token)?;
            self.context.apply_operator((ExprOperator::Plus, span));
        }
        Ok(token)
    }

    fn parse_atomic_expr(&mut self, token: <T::Iter as Iterator>::Item) -> ExprParsingResult<T> {
        match token {
            (Token::Eof, _) => {
                let span = token.1.clone();
                Err((token, InternalDiagnostic::new(Message::UnexpectedEof, span)))
            }
            (Token::Ident(ident), span) => {
                self.context.push_atom((ExprAtom::Ident(ident), span));
                Ok(self.bump())
            }
            (Token::Literal(literal), span) => {
                self.context.push_atom((ExprAtom::Literal(literal), span));
                Ok(self.bump())
            }
            (_, span) => Err((
                self.bump(),
                InternalDiagnostic::new(
                    Message::UnexpectedToken {
                        token: span.clone(),
                    },
                    span,
                ),
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
        actions: Vec<FileAction<SymSpan>>,
    }

    impl FileActionCollector {
        fn new() -> FileActionCollector {
            FileActionCollector {
                actions: Vec::new(),
            }
        }
    }

    impl DiagnosticsListener<SymSpan> for FileActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymSpan>) {
            self.actions.push(FileAction::EmitDiagnostic(diagnostic))
        }
    }

    impl FileContext<SymIdent, SymCommand, SymLiteral, SymSpan> for FileActionCollector {
        type StmtContext = StmtActionCollector;

        fn enter_stmt(self, label: Option<(SymIdent, SymSpan)>) -> StmtActionCollector {
            StmtActionCollector {
                label,
                actions: Vec::new(),
                parent: self,
            }
        }
    }

    struct StmtActionCollector {
        label: Option<(SymIdent, SymSpan)>,
        actions: Vec<StmtAction<SymSpan>>,
        parent: FileActionCollector,
    }

    impl DiagnosticsListener<SymSpan> for StmtActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymSpan>) {
            self.actions.push(StmtAction::EmitDiagnostic(diagnostic))
        }
    }

    impl StmtContext<SymIdent, SymCommand, SymLiteral, SymSpan> for StmtActionCollector {
        type CommandContext = CommandActionCollector;
        type MacroParamsContext = MacroParamsActionCollector;
        type MacroInvocationContext = MacroInvocationActionCollector;
        type Parent = FileActionCollector;

        fn enter_command(self, command: (SymCommand, SymSpan)) -> CommandActionCollector {
            CommandActionCollector {
                command,
                actions: Vec::new(),
                parent: self,
            }
        }

        fn enter_macro_def(self, keyword: SymSpan) -> MacroParamsActionCollector {
            MacroParamsActionCollector {
                keyword,
                actions: Vec::new(),
                parent: self,
            }
        }

        fn enter_macro_invocation(
            self,
            name: (SymIdent, SymSpan),
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
        command: (SymCommand, SymSpan),
        actions: Vec<CommandAction<SymSpan>>,
        parent: StmtActionCollector,
    }

    impl DiagnosticsListener<SymSpan> for CommandActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymSpan>) {
            self.actions.push(CommandAction::EmitDiagnostic(diagnostic))
        }
    }

    impl CommandContext<SymSpan> for CommandActionCollector {
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

    impl DiagnosticsListener<SymSpan> for ArgActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymSpan>) {
            self.expr_action_collector.emit_diagnostic(diagnostic)
        }
    }

    impl ExprContext<SymSpan> for ArgActionCollector {
        type Ident = SymIdent;
        type Literal = SymLiteral;
        type Parent = CommandActionCollector;

        fn push_atom(&mut self, atom: (ExprAtom<SymIdent, SymLiteral>, SymSpan)) {
            self.expr_action_collector.push_atom(atom)
        }

        fn apply_operator(&mut self, operator: (ExprOperator, SymSpan)) {
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
        actions: Vec<ExprAction<SymSpan>>,
    }

    impl ExprActionCollector {
        fn new() -> ExprActionCollector {
            ExprActionCollector {
                actions: Vec::new(),
            }
        }
    }

    impl DiagnosticsListener<SymSpan> for ExprActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymSpan>) {
            self.actions.push(ExprAction::EmitDiagnostic(diagnostic))
        }
    }

    impl ExprContext<SymSpan> for ExprActionCollector {
        type Ident = SymIdent;
        type Literal = SymLiteral;
        type Parent = Vec<ExprAction<SymSpan>>;

        fn push_atom(&mut self, atom: (ExprAtom<SymIdent, SymLiteral>, SymSpan)) {
            self.actions.push(ExprAction::PushAtom(atom))
        }

        fn apply_operator(&mut self, operator: (ExprOperator, SymSpan)) {
            self.actions.push(ExprAction::ApplyOperator(operator))
        }

        fn exit(self) -> Self::Parent {
            self.actions
        }
    }

    struct MacroParamsActionCollector {
        keyword: SymSpan,
        actions: Vec<MacroParamsAction<SymSpan>>,
        parent: StmtActionCollector,
    }

    impl DiagnosticsListener<SymSpan> for MacroParamsActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymSpan>) {
            self.actions
                .push(MacroParamsAction::EmitDiagnostic(diagnostic))
        }
    }

    impl MacroParamsContext<SymSpan> for MacroParamsActionCollector {
        type Command = SymCommand;
        type Ident = SymIdent;
        type Literal = SymLiteral;
        type MacroBodyContext = MacroBodyActionCollector;
        type Parent = StmtActionCollector;

        fn add_parameter(&mut self, param: (SymIdent, SymSpan)) {
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
        actions: Vec<TokenSeqAction<SymSpan>>,
        parent: MacroParamsActionCollector,
    }

    impl DiagnosticsListener<SymSpan> for MacroBodyActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymSpan>) {
            self.actions
                .push(TokenSeqAction::EmitDiagnostic(diagnostic))
        }
    }

    impl TokenSeqContext<SymSpan> for MacroBodyActionCollector {
        type Token = Token<SymIdent, SymCommand, SymLiteral>;
        type Parent = StmtActionCollector;

        fn push_token(&mut self, token: (Self::Token, SymSpan)) {
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
        name: (SymIdent, SymSpan),
        actions: Vec<MacroInvocationAction<SymSpan>>,
        parent: StmtActionCollector,
    }

    impl DiagnosticsListener<SymSpan> for MacroInvocationActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymSpan>) {
            self.actions
                .push(MacroInvocationAction::EmitDiagnostic(diagnostic))
        }
    }

    impl MacroInvocationContext<SymSpan> for MacroInvocationActionCollector {
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
        actions: Vec<TokenSeqAction<SymSpan>>,
        parent: MacroInvocationActionCollector,
    }

    impl DiagnosticsListener<SymSpan> for MacroArgActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SymSpan>) {
            self.actions
                .push(TokenSeqAction::EmitDiagnostic(diagnostic))
        }
    }

    impl TokenSeqContext<SymSpan> for MacroArgActionCollector {
        type Token = Token<SymIdent, SymCommand, SymLiteral>;
        type Parent = MacroInvocationActionCollector;

        fn push_token(&mut self, token: (Self::Token, SymSpan)) {
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

    fn assert_eq_actions(input: InputTokens, expected: impl Borrow<[FileAction<SymSpan>]>) {
        let mut parsing_context = FileActionCollector::new();
        parsing_context = parse_src(with_spans(&input.tokens), parsing_context);
        assert_eq!(parsing_context.actions, expected.borrow())
    }

    fn with_spans<'a>(
        tokens: impl IntoIterator<Item = &'a (SymToken, TokenRef)>,
    ) -> impl Iterator<Item = (SymToken, SymSpan)> {
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

    fn assert_eq_expr_diagnostics(mut input: InputTokens, expected: InternalDiagnostic<SymSpan>) {
        let expr_actions = parse_sym_expr(&mut input);
        assert_eq!(expr_actions, [ExprAction::EmitDiagnostic(expected)])
    }

    fn parse_sym_expr(input: &mut InputTokens) -> Vec<ExprAction<SymSpan>> {
        let tokens = &mut with_spans(&input.tokens);
        let mut parser = ExprParser {
            tokens,
            context: ExprActionCollector::new(),
        };
        let token = parser.bump();
        parser.parse(token).context
    }

    #[test]
    fn diagnose_stmt_starting_with_literal() {
        let token: SymSpan = TokenRef::from("a").into();
        assert_eq_actions(
            input_tokens![a @ Literal(())],
            [unlabeled(stmt_error(
                Message::UnexpectedToken { token },
                "a",
            ))],
        )
    }

    #[test]
    fn diagnose_missing_comma_in_arg_list() {
        let span: SymSpan = TokenRef::from("unexpected").into();
        assert_eq_actions(
            input_tokens![Command(()), Literal(()), unexpected @ Literal(())],
            [unlabeled(malformed_command(
                0,
                [expr().literal(1)],
                InternalDiagnostic::new(
                    Message::UnexpectedToken {
                        token: span.clone(),
                    },
                    span,
                ),
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
                arg_error(Message::UnexpectedEof, "eof"),
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
                arg_error(Message::UnexpectedEof, "eof"),
            ))],
        )
    }

    #[test]
    fn diagnose_unexpected_token_in_expr() {
        let input = input_tokens![plus @ Plus];
        let span: SymSpan = TokenRef::from("plus").into();
        assert_eq_expr_diagnostics(
            input,
            InternalDiagnostic::new(
                Message::UnexpectedToken {
                    token: span.clone(),
                },
                span,
            ),
        )
    }

    #[test]
    fn diagnose_unmatched_parentheses() {
        assert_eq_actions(
            input_tokens![Command(()), paren @ OpeningParenthesis, Literal(())],
            [unlabeled(command(
                0,
                [expr()
                    .literal(2)
                    .error(Message::UnmatchedParenthesis, TokenRef::from("paren"))],
            ))],
        )
    }

    #[test]
    fn diagnose_eof_for_rhs_operand() {
        assert_eq_rpn_expr(
            input_tokens![Ident(()), Plus],
            expr()
                .ident(0)
                .error(Message::UnexpectedEof, TokenRef::from(2)),
        )
    }

    #[test]
    fn recover_from_unexpected_token_in_expr() {
        let paren_span: SymSpan = TokenRef::from("paren").into();
        assert_eq_actions(
            input_tokens![
                Command(()),
                paren @ ClosingParenthesis,
                Plus,
                Ident(()),
                Eol,
                nop @ Command(())
            ],
            [
                unlabeled(command(
                    0,
                    [expr().error(
                        Message::UnexpectedToken {
                            token: paren_span.clone(),
                        },
                        paren_span,
                    )],
                )),
                unlabeled(command("nop", [])),
            ],
        )
    }
}
