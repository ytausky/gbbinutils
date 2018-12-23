use super::*;
use crate::diagnostics::span::{MergeSpans, StripSpan};
use crate::diagnostics::{CompactDiagnostic, EmitDiagnostic, Message};

type TokenKind = Token<(), (), (), ()>;

impl Copy for TokenKind {}

impl<C, I, L, E> Token<C, I, L, E> {
    fn kind(&self) -> TokenKind {
        use self::Token::*;
        match *self {
            ClosingParenthesis => ClosingParenthesis,
            Comma => Comma,
            Command(_) => Command(()),
            Endm => Endm,
            Eof => Eof,
            Eol => Eol,
            Error(_) => Error(()),
            Ident(_) => Ident(()),
            Label(_) => Label(()),
            Literal(_) => Literal(()),
            Macro => Macro,
            OpeningParenthesis => OpeningParenthesis,
            Plus => Plus,
        }
    }
}

const LINE_FOLLOW_SET: &[TokenKind] = &[Token::Eol, Token::Eof];

pub fn parse_src<Id, C, L, I, F, S>(mut tokens: I, context: F) -> F
where
    I: Iterator<Item = (Token<Id, C, L>, S)>,
    F: FileContext<Id, C, L, S>,
    S: Clone,
{
    let Parser { token, context, .. } = Parser::new(&mut tokens, context).parse_file();
    assert_eq!(token.0.kind(), Token::Eof);
    context
}

struct Parser<'a, T, I: 'a, C> {
    token: T,
    remaining: &'a mut I,
    context: C,
}

impl<'a, T, I: Iterator<Item = T>, C> Parser<'a, T, I, C> {
    fn new(tokens: &'a mut I, context: C) -> Self {
        Parser {
            token: tokens.next().unwrap(),
            remaining: tokens,
            context,
        }
    }
}

macro_rules! bump {
    ($parser:expr) => {
        $parser.token = $parser.remaining.next().unwrap()
    };
}

impl<'a, T, I, C> Parser<'a, T, I, C> {
    fn change_context<D, F: FnOnce(C) -> D>(self, f: F) -> Parser<'a, T, I, D> {
        Parser {
            token: self.token,
            remaining: self.remaining,
            context: f(self.context),
        }
    }
}

impl<'a, Id, C, L, S, I, A> Parser<'a, (Token<Id, C, L>, S), I, A> {
    fn token_is_in(&self, kinds: &[TokenKind]) -> bool {
        kinds.iter().any(|x| *x == self.token.0.kind())
    }
}

impl<'a, Id, C, L, I, Ctx, S> Parser<'a, (Token<Id, C, L>, S), I, Ctx>
where
    I: Iterator<Item = (Token<Id, C, L>, S)>,
    Ctx: FileContext<Id, C, L, S>,
    S: Clone,
{
    fn parse_file(mut self) -> Self {
        while self.token.0.kind() != Token::Eof {
            self = self.parse_stmt()
        }
        self
    }

    fn parse_stmt(mut self) -> Self {
        let label = if let (Token::Label(label), span) = self.token {
            bump!(self);
            Some((label, span))
        } else {
            None
        };
        self.change_context(|c| c.enter_stmt(label))
            .parse_unlabeled_stmt()
            .change_context(|c| c.exit())
    }
}

impl<'a, Id, C, L, I, Ctx, S> Parser<'a, (Token<Id, C, L>, S), I, Ctx>
where
    I: Iterator<Item = (Token<Id, C, L>, S)>,
    Ctx: StmtContext<Id, C, L, S>,
    S: Clone,
{
    fn parse_unlabeled_stmt(mut self) -> Self {
        loop {
            return match self.token {
                (Token::Eol, _) => {
                    bump!(self);
                    continue;
                }
                (Token::Label(_), _) | (Token::Eof, _) => self,
                (Token::Command(command), span) => {
                    bump!(self);
                    self.parse_command((command, span))
                }
                (Token::Ident(ident), span) => {
                    bump!(self);
                    self.parse_macro_invocation((ident, span))
                }
                (Token::Macro, span) => {
                    bump!(self);
                    self.parse_macro_def(span)
                }
                (_, span) => {
                    bump!(self);
                    let stripped = self.context.diagnostics().strip_span(&span);
                    self.context
                        .diagnostics()
                        .emit_diagnostic(CompactDiagnostic::new(
                            Message::UnexpectedToken { token: stripped },
                            span,
                        ));
                    self
                }
            };
        }
    }

    fn parse_command(self, command: (C, S)) -> Self {
        self.change_context(|c| c.enter_command(command))
            .parse_argument_list()
            .change_context(|c| c.exit())
    }

    fn parse_macro_def(self, span: S) -> Self {
        let mut state = self
            .change_context(|c| c.enter_macro_def(span))
            .parse_terminated_list(Token::Comma, LINE_FOLLOW_SET, |p| p.parse_macro_param());
        if state.token.0.kind() == Token::Eol {
            bump!(state);
            let mut state = state.change_context(|c| c.exit());
            loop {
                match state.token {
                    (Token::Endm, _) => {
                        state
                            .context
                            .push_token((Token::Eof, state.token.1.clone()));
                        bump!(state);
                        break;
                    }
                    (Token::Eof, _) => {
                        state
                            .context
                            .diagnostics()
                            .emit_diagnostic(CompactDiagnostic::new(
                                Message::UnexpectedEof,
                                state.token.1.clone(),
                            ));
                        break;
                    }
                    other => {
                        state.context.push_token(other);
                        bump!(state);
                    }
                }
            }
            state
        } else {
            assert_eq!(state.token.0.kind(), Token::Eof);
            state
                .context
                .diagnostics()
                .emit_diagnostic(CompactDiagnostic::new(
                    Message::UnexpectedEof,
                    state.token.1.clone(),
                ));
            state.change_context(|c| c.exit())
        }
        .change_context(|c| c.exit())
    }

    fn parse_macro_invocation(self, name: (Id, S)) -> Self {
        self.change_context(|c| c.enter_macro_invocation(name))
            .parse_macro_arg_list()
            .change_context(|c| c.exit())
    }
}

impl<'a, Id, C, L, I, Ctx, S> Parser<'a, (Token<Id, C, L>, S), I, Ctx>
where
    I: Iterator<Item = (Token<Id, C, L>, S)>,
    Ctx: CommandContext<S, Command = C, Ident = Id, Literal = L>,
    S: Clone,
{
    fn parse_argument_list(self) -> Self {
        self.parse_terminated_list(Token::Comma, LINE_FOLLOW_SET, |p| p.parse_argument())
    }

    fn parse_argument(self) -> Self {
        self.change_context(|c| c.add_argument())
            .parse()
            .change_context(|c| c.exit())
    }
}

type ParserResult<P, C, S> = Result<
    P,
    (
        P,
        ExpandedExprParsingError<<C as DelegateDiagnostics<S>>::Delegate, S>,
    ),
>;

type ExpandedExprParsingError<D, S> = ExprParsingError<S, <D as StripSpan<S>>::Stripped>;

enum ExprParsingError<S, R> {
    NothingParsed,
    Other(CompactDiagnostic<S, R>),
}

impl<'a, Id, C, L, I, Ctx, S> Parser<'a, (Token<Id, C, L>, S), I, Ctx>
where
    I: Iterator<Item = (Token<Id, C, L>, S)>,
    Ctx: ExprContext<S, Ident = Id, Literal = L>,
    S: Clone,
{
    fn parse(self) -> Self {
        self.parse_expression()
            .unwrap_or_else(|(mut parser, error)| {
                let diagnostic = match error {
                    ExprParsingError::NothingParsed => CompactDiagnostic::new(
                        match parser.token.0 {
                            Token::Eof => Message::UnexpectedEof,
                            _ => Message::UnexpectedToken {
                                token: parser.context.diagnostics().strip_span(&parser.token.1),
                            },
                        },
                        parser.token.1.clone(),
                    ),
                    ExprParsingError::Other(diagnostic) => diagnostic,
                };
                parser.context.diagnostics().emit_diagnostic(diagnostic);
                while !parser.token_is_in(LINE_FOLLOW_SET) {
                    bump!(parser);
                }
                parser
            })
    }

    fn parse_expression(mut self) -> ParserResult<Self, Ctx, S> {
        match self.token {
            (Token::OpeningParenthesis, span) => {
                bump!(self);
                self.parse_parenthesized_expression(span)
            }
            _ => self.parse_infix_expr(),
        }
    }

    fn parse_parenthesized_expression(mut self, left: S) -> ParserResult<Self, Ctx, S> {
        self = match self.parse_expression() {
            Ok(parser) => parser,
            Err((parser, error)) => {
                let error = match error {
                    error @ ExprParsingError::NothingParsed => match parser.token.0 {
                        Token::Eof | Token::Eol => ExprParsingError::Other(CompactDiagnostic::new(
                            Message::UnmatchedParenthesis,
                            left,
                        )),
                        _ => error,
                    },
                    error => error,
                };
                return Err((parser, error));
            }
        };
        match self.token {
            (Token::ClosingParenthesis, right) => {
                bump!(self);
                let span = self.context.diagnostics().merge_spans(&left, &right);
                self.context
                    .apply_operator((ExprOperator::Parentheses, span));
                Ok(self)
            }
            _ => Err((
                self,
                ExprParsingError::Other(CompactDiagnostic::new(
                    Message::UnmatchedParenthesis,
                    left,
                )),
            )),
        }
    }

    fn parse_infix_expr(mut self) -> ParserResult<Self, Ctx, S> {
        self = self.parse_atomic_expr()?;
        while let (Token::Plus, span) = self.token {
            bump!(self);
            self = self.parse_atomic_expr()?;
            self.context.apply_operator((ExprOperator::Plus, span));
        }
        Ok(self)
    }

    fn parse_atomic_expr(mut self) -> ParserResult<Self, Ctx, S> {
        match self.token.0 {
            Token::Eof | Token::Eol => Err((self, ExprParsingError::NothingParsed)),
            Token::Ident(ident) => {
                self.context
                    .push_atom((ExprAtom::Ident(ident), self.token.1));
                bump!(self);
                Ok(self)
            }
            Token::Literal(literal) => {
                self.context
                    .push_atom((ExprAtom::Literal(literal), self.token.1));
                bump!(self);
                Ok(self)
            }
            _ => {
                let span = self.token.1;
                let stripped = self.context.diagnostics().strip_span(&span);
                bump!(self);
                Err((
                    self,
                    ExprParsingError::Other(CompactDiagnostic::new(
                        Message::UnexpectedToken { token: stripped },
                        span,
                    )),
                ))
            }
        }
    }
}

impl<'a, Id, C, L, I, Ctx, S> Parser<'a, (Token<Id, C, L>, S), I, Ctx>
where
    I: Iterator<Item = (Token<Id, C, L>, S)>,
    Ctx: MacroParamsContext<S, Command = C, Ident = Id, Literal = L>,
    S: Clone,
{
    fn parse_macro_param(mut self) -> Self {
        match self.token.0 {
            Token::Ident(ident) => self.context.add_parameter((ident, self.token.1)),
            _ => {
                let stripped = self.context.diagnostics().strip_span(&self.token.1);
                self.context
                    .diagnostics()
                    .emit_diagnostic(CompactDiagnostic::new(
                        Message::UnexpectedToken { token: stripped },
                        self.token.1.clone(),
                    ))
            }
        };
        bump!(self);
        self
    }
}

impl<'a, Id, C, L, I, Ctx, S> Parser<'a, (Token<Id, C, L>, S), I, Ctx>
where
    I: Iterator<Item = (Token<Id, C, L>, S)>,
    Ctx: MacroInvocationContext<S, Token = Token<Id, C, L>>,
    S: Clone,
{
    fn parse_macro_arg_list(self) -> Self {
        self.parse_terminated_list(Token::Comma, LINE_FOLLOW_SET, |p| {
            let mut state = p.change_context(|c| c.enter_macro_arg());
            loop {
                match state.token {
                    (Token::Comma, _) | (Token::Eol, _) | (Token::Eof, _) => break,
                    other => {
                        bump!(state);
                        state.context.push_token(other)
                    }
                }
            }
            state.change_context(|c| c.exit())
        })
    }
}

impl<'a, Id, C, L, I, Ctx, S> Parser<'a, (Token<Id, C, L>, S), I, Ctx>
where
    I: Iterator<Item = (Token<Id, C, L>, S)>,
    Ctx: DelegateDiagnostics<S>,
    S: Clone,
{
    fn parse_terminated_list<P>(
        mut self,
        delimiter: TokenKind,
        terminators: &[TokenKind],
        parser: P,
    ) -> Self
    where
        P: FnMut(Self) -> Self,
    {
        self = self.parse_list(delimiter, terminators, parser);
        if !self.token_is_in(terminators) {
            let unexpected_span = self.token.1;
            let stripped = self.context.diagnostics().strip_span(&unexpected_span);
            self.context
                .diagnostics()
                .emit_diagnostic(CompactDiagnostic::new(
                    Message::UnexpectedToken { token: stripped },
                    unexpected_span,
                ));
            bump!(self);
            while !self.token_is_in(terminators) {
                bump!(self);
            }
        }
        self
    }

    fn parse_list<P>(self, delimiter: TokenKind, terminators: &[TokenKind], mut parser: P) -> Self
    where
        P: FnMut(Self) -> Self,
    {
        if self.token_is_in(terminators) {
            self
        } else {
            self.parse_nonempty_list(delimiter, &mut parser)
        }
    }

    fn parse_nonempty_list<P>(mut self, delimiter: TokenKind, parser: &mut P) -> Self
    where
        P: FnMut(Self) -> Self,
    {
        self = parser(self);
        while self.token.0.kind() == delimiter {
            bump!(self);
            self = parser(self);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::ast::*;
    use super::Token::*;
    use super::*;
    use crate::diagnostics::span::{MergeSpans, StripSpan};
    use crate::diagnostics::{CompactDiagnostic, EmitDiagnostic, Message};
    use crate::frontend::syntax::{ExprAtom, ExprOperator};
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

    impl MergeSpans<SymSpan> for FileActionCollector {
        fn merge_spans(&mut self, left: &SymSpan, right: &SymSpan) -> SymSpan {
            SymSpan::merge(left, right)
        }
    }

    impl StripSpan<SymSpan> for FileActionCollector {
        type Stripped = SymSpan;

        fn strip_span(&mut self, span: &SymSpan) -> Self::Stripped {
            span.clone()
        }
    }

    impl EmitDiagnostic<SymSpan, SymSpan> for FileActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<SymSpan, SymSpan>) {
            self.actions.push(FileAction::EmitDiagnostic(diagnostic))
        }
    }

    impl DelegateDiagnostics<SymSpan> for FileActionCollector {
        type Delegate = Self;

        fn diagnostics(&mut self) -> &mut Self::Delegate {
            self
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

    impl MergeSpans<SymSpan> for StmtActionCollector {
        fn merge_spans(&mut self, left: &SymSpan, right: &SymSpan) -> SymSpan {
            SymSpan::merge(left, right)
        }
    }

    impl StripSpan<SymSpan> for StmtActionCollector {
        type Stripped = SymSpan;

        fn strip_span(&mut self, span: &SymSpan) -> Self::Stripped {
            span.clone()
        }
    }

    impl EmitDiagnostic<SymSpan, SymSpan> for StmtActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<SymSpan, SymSpan>) {
            self.actions.push(StmtAction::EmitDiagnostic(diagnostic))
        }
    }

    impl DelegateDiagnostics<SymSpan> for StmtActionCollector {
        type Delegate = Self;

        fn diagnostics(&mut self) -> &mut Self::Delegate {
            self
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

    impl MergeSpans<SymSpan> for CommandActionCollector {
        fn merge_spans(&mut self, left: &SymSpan, right: &SymSpan) -> SymSpan {
            SymSpan::merge(left, right)
        }
    }

    impl StripSpan<SymSpan> for CommandActionCollector {
        type Stripped = SymSpan;

        fn strip_span(&mut self, span: &SymSpan) -> Self::Stripped {
            span.clone()
        }
    }

    impl EmitDiagnostic<SymSpan, SymSpan> for CommandActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<SymSpan, SymSpan>) {
            self.actions.push(CommandAction::EmitDiagnostic(diagnostic))
        }
    }

    impl DelegateDiagnostics<SymSpan> for CommandActionCollector {
        type Delegate = Self;

        fn diagnostics(&mut self) -> &mut Self::Delegate {
            self
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

    impl MergeSpans<SymSpan> for ArgActionCollector {
        fn merge_spans(&mut self, left: &SymSpan, right: &SymSpan) -> SymSpan {
            SymSpan::merge(left, right)
        }
    }

    impl StripSpan<SymSpan> for ArgActionCollector {
        type Stripped = SymSpan;

        fn strip_span(&mut self, span: &SymSpan) -> Self::Stripped {
            span.clone()
        }
    }

    impl EmitDiagnostic<SymSpan, SymSpan> for ArgActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<SymSpan, SymSpan>) {
            self.expr_action_collector.emit_diagnostic(diagnostic)
        }
    }

    impl DelegateDiagnostics<SymSpan> for ArgActionCollector {
        type Delegate = Self;

        fn diagnostics(&mut self) -> &mut Self::Delegate {
            self
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

    impl MergeSpans<SymSpan> for ExprActionCollector {
        fn merge_spans(&mut self, left: &SymSpan, right: &SymSpan) -> SymSpan {
            SymSpan::merge(left, right)
        }
    }

    impl StripSpan<SymSpan> for ExprActionCollector {
        type Stripped = SymSpan;

        fn strip_span(&mut self, span: &SymSpan) -> Self::Stripped {
            span.clone()
        }
    }

    impl EmitDiagnostic<SymSpan, SymSpan> for ExprActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<SymSpan, SymSpan>) {
            self.actions.push(ExprAction::EmitDiagnostic(diagnostic))
        }
    }

    impl DelegateDiagnostics<SymSpan> for ExprActionCollector {
        type Delegate = Self;

        fn diagnostics(&mut self) -> &mut Self::Delegate {
            self
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

    impl MergeSpans<SymSpan> for MacroParamsActionCollector {
        fn merge_spans(&mut self, left: &SymSpan, right: &SymSpan) -> SymSpan {
            SymSpan::merge(left, right)
        }
    }

    impl StripSpan<SymSpan> for MacroParamsActionCollector {
        type Stripped = SymSpan;

        fn strip_span(&mut self, span: &SymSpan) -> Self::Stripped {
            span.clone()
        }
    }

    impl EmitDiagnostic<SymSpan, SymSpan> for MacroParamsActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<SymSpan, SymSpan>) {
            self.actions
                .push(MacroParamsAction::EmitDiagnostic(diagnostic))
        }
    }

    impl DelegateDiagnostics<SymSpan> for MacroParamsActionCollector {
        type Delegate = Self;

        fn diagnostics(&mut self) -> &mut Self::Delegate {
            self
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

    impl MergeSpans<SymSpan> for MacroBodyActionCollector {
        fn merge_spans(&mut self, left: &SymSpan, right: &SymSpan) -> SymSpan {
            SymSpan::merge(left, right)
        }
    }

    impl StripSpan<SymSpan> for MacroBodyActionCollector {
        type Stripped = SymSpan;

        fn strip_span(&mut self, span: &SymSpan) -> Self::Stripped {
            span.clone()
        }
    }

    impl EmitDiagnostic<SymSpan, SymSpan> for MacroBodyActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<SymSpan, SymSpan>) {
            self.actions
                .push(TokenSeqAction::EmitDiagnostic(diagnostic))
        }
    }

    impl DelegateDiagnostics<SymSpan> for MacroBodyActionCollector {
        type Delegate = Self;

        fn diagnostics(&mut self) -> &mut Self::Delegate {
            self
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

    impl MergeSpans<SymSpan> for MacroInvocationActionCollector {
        fn merge_spans(&mut self, left: &SymSpan, right: &SymSpan) -> SymSpan {
            SymSpan::merge(left, right)
        }
    }

    impl StripSpan<SymSpan> for MacroInvocationActionCollector {
        type Stripped = SymSpan;

        fn strip_span(&mut self, span: &SymSpan) -> Self::Stripped {
            span.clone()
        }
    }

    impl EmitDiagnostic<SymSpan, SymSpan> for MacroInvocationActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<SymSpan, SymSpan>) {
            self.actions
                .push(MacroInvocationAction::EmitDiagnostic(diagnostic))
        }
    }

    impl DelegateDiagnostics<SymSpan> for MacroInvocationActionCollector {
        type Delegate = Self;

        fn diagnostics(&mut self) -> &mut Self::Delegate {
            self
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

    impl MergeSpans<SymSpan> for MacroArgActionCollector {
        fn merge_spans(&mut self, left: &SymSpan, right: &SymSpan) -> SymSpan {
            SymSpan::merge(left, right)
        }
    }

    impl StripSpan<SymSpan> for MacroArgActionCollector {
        type Stripped = SymSpan;

        fn strip_span(&mut self, span: &SymSpan) -> Self::Stripped {
            span.clone()
        }
    }

    impl EmitDiagnostic<SymSpan, SymSpan> for MacroArgActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<SymSpan, SymSpan>) {
            self.actions
                .push(TokenSeqAction::EmitDiagnostic(diagnostic))
        }
    }

    impl DelegateDiagnostics<SymSpan> for MacroArgActionCollector {
        type Delegate = Self;

        fn diagnostics(&mut self) -> &mut Self::Delegate {
            self
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
        assert_eq_actions(input_tokens![Eol], [unlabeled(empty())])
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
            [unlabeled(command("nop", []))],
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
            unlabeled(command(6, [expr().ident(7), expr().literal(9)])),
        ];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_empty_macro_definition() {
        let tokens = input_tokens![Label(()), Macro, Eol, Endm];
        let expected_actions = [labeled(0, macro_def(1, [], Vec::new(), 3))];
        assert_eq_actions(tokens, expected_actions);
    }

    #[test]
    fn parse_macro_definition_with_instruction() {
        let tokens = input_tokens![Label(()), Macro, Eol, Command(()), Eol, Endm];
        let expected_actions = [labeled(0, macro_def(1, [], tokens.token_seq([3, 4]), 5))];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_nonempty_macro_def_with_two_params() {
        let tokens = input_tokens![
            Label(()),
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
            macro_def(1, [2.into(), 4.into()], tokens.token_seq([6, 7]), 8),
        )];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_label() {
        let tokens = input_tokens![Label(()), Eol];
        let expected_actions = [labeled(0, empty())];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_two_consecutive_labels() {
        let tokens = input_tokens![Label(()), Eol, Label(())];
        let expected = [labeled(0, empty()), labeled(2, empty())];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_labeled_instruction() {
        let tokens = input_tokens![Label(()), Command(()), Eol];
        let expected = [labeled(0, command(1, [])), unlabeled(empty())];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_labeled_command_with_eol_separators() {
        let tokens = input_tokens![Label(()), Eol, Eol, Command(())];
        let expected = [labeled(0, command(3, []))];
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
        expected: CompactDiagnostic<SymSpan, SymSpan>,
    ) {
        let expr_actions = parse_sym_expr(&mut input);
        assert_eq!(expr_actions, [ExprAction::EmitDiagnostic(expected)])
    }

    fn parse_sym_expr(input: &mut InputTokens) -> Vec<ExprAction<SymSpan>> {
        let tokens = &mut with_spans(&input.tokens);
        Parser::new(tokens, ExprActionCollector::new())
            .parse()
            .change_context(|c| c.exit())
            .context
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
                CompactDiagnostic::new(
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
            CompactDiagnostic::new(
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

    #[test]
    fn diagnose_unmatched_parenthesis_at_eol() {
        assert_eq_actions(
            input_tokens![Command(()), OpeningParenthesis, Eol],
            [
                unlabeled(command(
                    0,
                    [expr().error(Message::UnmatchedParenthesis, TokenRef::from(1))],
                )),
                unlabeled(empty()),
            ],
        )
    }

    #[test]
    fn diagnose_unexpected_token_in_macro_param() {
        let span: SymSpan = TokenRef::from(1).into();
        assert_eq_actions(
            input_tokens![Macro, Literal(()), Eol, Endm],
            [unlabeled(vec![StmtAction::MacroDef {
                keyword: TokenRef::from(0).into(),
                params: vec![MacroParamsAction::EmitDiagnostic(CompactDiagnostic::new(
                    Message::UnexpectedToken {
                        token: span.clone(),
                    },
                    span,
                ))],
                body: vec![TokenSeqAction::PushToken((Eof, TokenRef::from(3).into()))],
            }])],
        )
    }
}
