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
    let token = parser.bump();
    let State { token, context } = parser.parse_file(State { token, context });
    assert_eq!(token.0.kind(), Token::Eof);
    context
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

impl<Id, C, L, S: Span, I: Iterator<Item = (Token<Id, C, L>, S)>> Parser<I> {
    fn parse_file<F: FileContext<Id, C, L, S>>(
        &mut self,
        state: State<I::Item, F>,
    ) -> State<I::Item, F> {
        self.parse_terminated_list(Token::Eol, &[Token::Eof], |p, s| p.parse_stmt(s), state)
    }

    fn parse_stmt<F: FileContext<Id, C, L, S>>(
        &mut self,
        mut state: State<I::Item, F>,
    ) -> State<I::Item, F> {
        match state.token {
            (Token::Ident(ident), span) => {
                state.token = self.bump();
                self.parse_potentially_labeled_stmt((ident, span), state)
            }
            _ => self
                .parse_unlabeled_stmt(state.change_context(|c| c.enter_stmt(None)))
                .change_context(|c| c.exit()),
        }
    }

    fn parse_potentially_labeled_stmt<F: FileContext<Id, C, L, S>>(
        &mut self,
        ident: (Id, S),
        mut state: State<I::Item, F>,
    ) -> State<I::Item, F> {
        if let (Token::Colon, _) = state.token {
            state.token = self.bump();
            self.parse_unlabeled_stmt(state.change_context(|c| c.enter_stmt(Some(ident))))
        } else {
            self.parse_macro_invocation(ident, state.change_context(|c| c.enter_stmt(None)))
        }.change_context(|c| c.exit())
    }

    fn parse_unlabeled_stmt<LA: StmtContext<Id, C, L, S>>(
        &mut self,
        mut state: State<I::Item, LA>,
    ) -> State<I::Item, LA> {
        match state.token {
            (Token::Eol, _) | (Token::Eof, _) => state,
            (Token::Command(command), span) => {
                state.token = self.bump();
                self.parse_command((command, span), state)
            }
            (Token::Ident(ident), span) => {
                state.token = self.bump();
                self.parse_macro_invocation((ident, span), state)
            }
            (Token::Macro, span) => {
                state.token = self.bump();
                self.parse_macro_def(span, state)
            }
            (_, span) => {
                state.token = self.bump();
                state.context.emit_diagnostic(InternalDiagnostic::new(
                    Message::UnexpectedToken {
                        token: span.clone(),
                    },
                    span,
                ));
                state
            }
        }
    }

    fn parse_macro_def<LA: StmtContext<Id, C, L, S>>(
        &mut self,
        span: S,
        state: State<I::Item, LA>,
    ) -> State<I::Item, LA> {
        let mut state = self.parse_terminated_list(
            Token::Comma,
            LINE_FOLLOW_SET,
            |p, s| p.parse_macro_param(s),
            state.change_context(|c| c.enter_macro_def(span)),
        );
        if state.token.0.kind() == Token::Eol {
            state.token = self.bump();
            let mut state = state.change_context(|c| c.exit());
            loop {
                match state.token {
                    (Token::Endm, _) => {
                        state
                            .context
                            .push_token((Token::Eof, state.token.1.clone()));
                        state.token = self.bump();
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
                        state.token = self.bump()
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

    fn parse_macro_param<MPA>(&mut self, mut state: State<I::Item, MPA>) -> State<I::Item, MPA>
    where
        MPA: MacroParamsContext<S, Command = C, Ident = Id, Literal = L>,
    {
        let ident = match state.token {
            (Token::Ident(ident), span) => (ident, span),
            _ => panic!(),
        };
        state.token = self.bump();
        state.context.add_parameter(ident);
        state
    }

    fn parse_command<LA: StmtContext<Id, C, L, S>>(
        &mut self,
        command: (C, S),
        state: State<I::Item, LA>,
    ) -> State<I::Item, LA> {
        self.parse_argument_list(state.change_context(|c| c.enter_command(command)))
            .change_context(|c| c.exit())
    }

    fn parse_macro_invocation<LA: StmtContext<Id, C, L, S>>(
        &mut self,
        name: (Id, S),
        state: State<I::Item, LA>,
    ) -> State<I::Item, LA> {
        self.parse_macro_arg_list(state.change_context(|c| c.enter_macro_invocation(name)))
            .change_context(|c| c.exit())
    }

    fn parse_argument_list<CC: CommandContext<S, Command = C, Ident = Id, Literal = L>>(
        &mut self,
        state: State<I::Item, CC>,
    ) -> State<I::Item, CC> {
        self.parse_terminated_list(
            Token::Comma,
            LINE_FOLLOW_SET,
            |p, s| p.parse_argument(s),
            state,
        )
    }

    fn parse_macro_arg_list<M: MacroInvocationContext<S, Token = Token<Id, C, L>>>(
        &mut self,
        state: State<I::Item, M>,
    ) -> State<I::Item, M> {
        self.parse_terminated_list(
            Token::Comma,
            LINE_FOLLOW_SET,
            |p, state| {
                let mut state = state.change_context(|c| c.enter_macro_arg());
                loop {
                    match state.token {
                        (Token::Comma, _) | (Token::Eol, _) | (Token::Eof, _) => break,
                        other => {
                            state.token = p.bump();
                            state.context.push_token(other)
                        }
                    }
                }
                state.change_context(|c| c.exit())
            },
            state,
        )
    }

    fn parse_terminated_list<P, A>(
        &mut self,
        delimiter: TokenKind,
        terminators: &[TokenKind],
        parser: P,
        mut state: State<I::Item, A>,
    ) -> State<I::Item, A>
    where
        P: FnMut(&mut Self, State<I::Item, A>) -> State<I::Item, A>,
        A: DiagnosticsListener<S>,
    {
        state = self.parse_list(delimiter, terminators, parser, state);
        if !state.token_is_in(terminators) {
            let unexpected_span = state.token.1;
            state.context.emit_diagnostic(InternalDiagnostic::new(
                Message::UnexpectedToken {
                    token: unexpected_span.clone(),
                },
                unexpected_span,
            ));
            state.token = self.bump();
            while !state.token_is_in(terminators) {
                state.token = self.bump();
            }
        }
        state
    }

    fn parse_list<P, A>(
        &mut self,
        delimiter: TokenKind,
        terminators: &[TokenKind],
        mut parser: P,
        state: State<I::Item, A>,
    ) -> State<I::Item, A>
    where
        P: FnMut(&mut Self, State<I::Item, A>) -> State<I::Item, A>,
        A: DiagnosticsListener<S>,
    {
        if state.token_is_in(terminators) {
            state
        } else {
            self.parse_nonempty_list(delimiter, &mut parser, state)
        }
    }

    fn parse_nonempty_list<P, A>(
        &mut self,
        delimiter: TokenKind,
        parser: &mut P,
        mut state: State<I::Item, A>,
    ) -> State<I::Item, A>
    where
        P: FnMut(&mut Self, State<I::Item, A>) -> State<I::Item, A>,
        A: DiagnosticsListener<S>,
    {
        state = parser(self, state);
        while state.token.0.kind() == delimiter {
            state.token = self.bump();
            state = parser(self, state);
        }
        state
    }

    fn parse_argument<CC: CommandContext<S, Command = C, Ident = Id, Literal = L>>(
        &mut self,
        state: State<I::Item, CC>,
    ) -> State<I::Item, CC> {
        ExprParser {
            tokens: self,
            context: state.context.add_argument(),
        }.parse(state.token)
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
    fn parse(
        mut self,
        token: <T::Iter as Iterator>::Item,
    ) -> State<<T::Iter as Iterator>::Item, A::Parent> {
        let (mut token, result) = self.parse_expression(token);
        if let Err(diagnostic) = result {
            self.context.emit_diagnostic(diagnostic);
            while !LINE_FOLLOW_SET.iter().any(|x| *x == token.0.kind()) {
                token = self.bump();
            }
        }
        State {
            token,
            context: self.context.exit(),
        }
    }

    fn parse_expression(
        &mut self,
        mut token: <T::Iter as Iterator>::Item,
    ) -> (
        <T::Iter as Iterator>::Item,
        Result<(), InternalDiagnostic<T::Span>>,
    ) {
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
    ) -> (
        <T::Iter as Iterator>::Item,
        Result<(), InternalDiagnostic<T::Span>>,
    ) {
        let (mut token, result) = self.parse_expression(token);
        if result.is_err() {
            return (token, result);
        }
        match token {
            (Token::ClosingParenthesis, right) => {
                token = self.bump();
                self.context
                    .apply_operator((ExprOperator::Parentheses, left.extend(&right)));
                (token, Ok(()))
            }
            other => (
                other,
                Err(InternalDiagnostic::new(Message::UnmatchedParenthesis, left)),
            ),
        }
    }

    fn parse_infix_expr(
        &mut self,
        token: <T::Iter as Iterator>::Item,
    ) -> (
        <T::Iter as Iterator>::Item,
        Result<(), InternalDiagnostic<T::Span>>,
    ) {
        let (mut token, result) = self.parse_atomic_expr(token);
        if result.is_err() {
            return (token, result);
        }
        loop {
            match token {
                (Token::Plus, span) => {
                    token = self.bump();
                    let (new_token, result) = self.parse_atomic_expr(token);
                    token = new_token;
                    if result.is_err() {
                        return (token, result);
                    }
                    self.context.apply_operator((ExprOperator::Plus, span));
                }
                _ => break,
            }
        }
        (token, Ok(()))
    }

    fn parse_atomic_expr(
        &mut self,
        token: <T::Iter as Iterator>::Item,
    ) -> (
        <T::Iter as Iterator>::Item,
        Result<(), InternalDiagnostic<T::Span>>,
    ) {
        match token {
            (Token::Eof, _) => {
                let span = token.1.clone();
                (
                    token,
                    Err(InternalDiagnostic::new(Message::UnexpectedEof, span)),
                )
            }
            (Token::Ident(ident), span) => {
                self.context.push_atom((ExprAtom::Ident(ident), span));
                (self.bump(), Ok(()))
            }
            (Token::Literal(literal), span) => {
                self.context.push_atom((ExprAtom::Literal(literal), span));
                (self.bump(), Ok(()))
            }
            (_, span) => (
                self.bump(),
                Err(InternalDiagnostic::new(
                    Message::UnexpectedToken {
                        token: span.clone(),
                    },
                    span,
                )),
            ),
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
        let tokens = &mut with_spans(&input.tokens).peekable();
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
