use super::SimpleToken::*;
use super::*;
use crate::diag::span::{MergeSpans, StripSpan};
use crate::diag::{CompactDiagnostic, EmitDiagnostic, Message};
use crate::expr::BinaryOperator;

type TokenKind = Token<(), (), ()>;

impl Copy for TokenKind {}

impl<I, L, C> Token<I, L, C> {
    fn kind(&self) -> TokenKind {
        use self::Token::*;
        match *self {
            Command(_) => Command(()),
            Ident(_) => Ident(()),
            Label(_) => Label(()),
            Literal(_) => Literal(()),
            Simple(simple) => Simple(simple),
        }
    }
}

const LINE_FOLLOW_SET: &[TokenKind] = &[Token::Simple(Eol), Token::Simple(Eof)];

pub(crate) fn parse_src<Id, L, C, E, I, F, S>(mut tokens: I, context: F) -> F
where
    I: Iterator<Item = (Result<Token<Id, L, C>, E>, S)>,
    F: FileContext<Id, L, C, S>,
    S: Clone,
{
    let Parser { token, context, .. } = Parser::new(&mut tokens, context).parse_file();
    assert_eq!(
        token.0.ok().as_ref().map(Token::kind),
        Some(Token::Simple(SimpleToken::Eof))
    );
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

impl<'a, Id, L, C, E, S, I, A> Parser<'a, (Result<Token<Id, L, C>, E>, S), I, A> {
    fn token_kind(&self) -> Option<TokenKind> {
        self.token.0.as_ref().ok().map(Token::kind)
    }

    fn token_is_in(&self, kinds: &[TokenKind]) -> bool {
        match &self.token.0 {
            Ok(token) => kinds.iter().any(|x| *x == token.kind()),
            Err(_) => false,
        }
    }
}

impl<'a, Id, L, C, E, I, Ctx, S> Parser<'a, (Result<Token<Id, L, C>, E>, S), I, Ctx>
where
    I: Iterator<Item = (Result<Token<Id, L, C>, E>, S)>,
    Ctx: FileContext<Id, L, C, S>,
    S: Clone,
{
    fn parse_file(mut self) -> Self {
        while self.token_kind() != Some(Token::Simple(SimpleToken::Eof)) {
            self = self.parse_stmt()
        }
        self
    }

    fn parse_stmt(mut self) -> Self {
        let label = if let (Ok(Token::Label(label)), span) = self.token {
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

impl<'a, Id, L, C, E, I, Ctx, S> Parser<'a, (Result<Token<Id, L, C>, E>, S), I, Ctx>
where
    I: Iterator<Item = (Result<Token<Id, L, C>, E>, S)>,
    Ctx: StmtContext<Id, L, C, S>,
    S: Clone,
{
    fn parse_unlabeled_stmt(mut self) -> Self {
        loop {
            return match self.token {
                (Ok(Token::Simple(SimpleToken::Eol)), _) => {
                    bump!(self);
                    continue;
                }
                (Ok(Token::Label(_)), _) | (Ok(Token::Simple(SimpleToken::Eof)), _) => self,
                (Ok(Token::Command(command)), span) => {
                    bump!(self);
                    self.parse_command((command, span))
                }
                (Ok(Token::Ident(ident)), span) => {
                    bump!(self);
                    self.parse_macro_invocation((ident, span))
                }
                (Ok(Token::Simple(SimpleToken::Macro)), span) => {
                    bump!(self);
                    self.parse_macro_def(span)
                }
                (_, span) => {
                    bump!(self);
                    let stripped = self.context.diagnostics().strip_span(&span);
                    self.context
                        .diagnostics()
                        .emit_diagnostic(Message::UnexpectedToken { token: stripped }.at(span));
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
            .parse_terminated_list(Comma.into(), LINE_FOLLOW_SET, |p| p.parse_macro_param());
        if state.token_kind() == Some(Eol.into()) {
            bump!(state);
            let mut state = state.change_context(|c| c.exit());
            loop {
                match state.token {
                    (Ok(Token::Simple(Endm)), _) => {
                        state
                            .context
                            .push_token((Eof.into(), state.token.1.clone()));
                        bump!(state);
                        break;
                    }
                    (Ok(Token::Simple(Eof)), _) => {
                        state
                            .context
                            .diagnostics()
                            .emit_diagnostic(Message::UnexpectedEof.at(state.token.1.clone()));
                        break;
                    }
                    (Ok(other), span) => {
                        state.context.push_token((other, span));
                        bump!(state);
                    }
                    (Err(_), _) => unimplemented!(),
                }
            }
            state
        } else {
            assert_eq!(state.token_kind(), Some(Eof.into()));
            state
                .context
                .diagnostics()
                .emit_diagnostic(Message::UnexpectedEof.at(state.token.1.clone()));
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

impl<'a, Id, L, C, E, I, Ctx, S> Parser<'a, (Result<Token<Id, L, C>, E>, S), I, Ctx>
where
    I: Iterator<Item = (Result<Token<Id, L, C>, E>, S)>,
    Ctx: CommandContext<S, Command = C, Ident = Id, Literal = L>,
    S: Clone,
{
    fn parse_argument_list(self) -> Self {
        self.parse_terminated_list(Comma.into(), LINE_FOLLOW_SET, |p| p.parse_argument())
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

enum SuffixOperator {
    Binary(BinaryOperator),
    FnCall,
}

impl<I, C, L> Token<I, C, L> {
    fn as_suffix_operator(&self) -> Option<SuffixOperator> {
        use SuffixOperator::*;
        match self {
            Token::Simple(Minus) => Some(Binary(BinaryOperator::Minus)),
            Token::Simple(OpeningParenthesis) => Some(FnCall),
            Token::Simple(Pipe) => Some(Binary(BinaryOperator::BitwiseOr)),
            Token::Simple(Plus) => Some(Binary(BinaryOperator::Plus)),
            Token::Simple(Slash) => Some(Binary(BinaryOperator::Division)),
            Token::Simple(Star) => Some(Binary(BinaryOperator::Multiplication)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
enum Precedence {
    None,
    BitwiseOr,
    Addition,
    Multiplication,
    FnCall,
}

impl SuffixOperator {
    fn precedence(&self) -> Precedence {
        use SuffixOperator::*;
        match self {
            Binary(BinaryOperator::BitwiseOr) => Precedence::BitwiseOr,
            Binary(BinaryOperator::Plus) | Binary(BinaryOperator::Minus) => Precedence::Addition,
            Binary(BinaryOperator::Multiplication) | Binary(BinaryOperator::Division) => {
                Precedence::Multiplication
            }
            FnCall => Precedence::FnCall,
        }
    }
}

impl<'a, Id, L, C, E, I, Ctx, S> Parser<'a, (Result<Token<Id, L, C>, E>, S), I, Ctx>
where
    I: Iterator<Item = (Result<Token<Id, L, C>, E>, S)>,
    Ctx: ExprContext<S, Ident = Id, Literal = L>,
    S: Clone,
{
    fn parse(self) -> Self {
        self.parse_expression()
            .unwrap_or_else(|(mut parser, error)| {
                let diagnostic = match error {
                    ExprParsingError::NothingParsed => match parser.token.0 {
                        Ok(Token::Simple(Eof)) => Message::UnexpectedEof,
                        _ => Message::UnexpectedToken {
                            token: parser.context.diagnostics().strip_span(&parser.token.1),
                        },
                    }
                    .at(parser.token.1.clone())
                    .into(),
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
            (Ok(Token::Simple(OpeningParenthesis)), span) => {
                bump!(self);
                self.parse_parenthesized_expression(span)
            }
            _ => self.parse_infix_expr(Precedence::None),
        }
    }

    fn parse_parenthesized_expression(mut self, left: S) -> ParserResult<Self, Ctx, S> {
        self = match self.parse_expression() {
            Ok(parser) => parser,
            Err((parser, error)) => {
                let error = match error {
                    error @ ExprParsingError::NothingParsed => match parser.token.0 {
                        Ok(Token::Simple(Eof)) | Ok(Token::Simple(Eol)) => {
                            ExprParsingError::Other(Message::UnmatchedParenthesis.at(left).into())
                        }
                        _ => error,
                    },
                    error => error,
                };
                return Err((parser, error));
            }
        };
        match self.token {
            (Ok(Token::Simple(ClosingParenthesis)), right) => {
                bump!(self);
                let span = self.context.diagnostics().merge_spans(&left, &right);
                self.context
                    .apply_operator((Operator::Unary(UnaryOperator::Parentheses), span));
                Ok(self)
            }
            _ => Err((
                self,
                ExprParsingError::Other(Message::UnmatchedParenthesis.at(left).into()),
            )),
        }
    }

    fn parse_infix_expr(mut self, lowest: Precedence) -> ParserResult<Self, Ctx, S> {
        self = self.parse_atomic_expr()?;
        while let Some(suffix_operator) = self
            .token
            .0
            .as_ref()
            .ok()
            .map(Token::as_suffix_operator)
            .unwrap_or(None)
        {
            let precedence = suffix_operator.precedence();
            if precedence <= lowest {
                break;
            }
            let span = self.token.1;
            bump!(self);
            match suffix_operator {
                SuffixOperator::Binary(binary_operator) => {
                    self = self.parse_infix_expr(precedence)?;
                    self.context
                        .apply_operator((Operator::Binary(binary_operator), span))
                }
                SuffixOperator::FnCall => self = self.parse_fn_call(span)?,
            }
        }
        Ok(self)
    }

    fn parse_fn_call(mut self, left: S) -> ParserResult<Self, Ctx, S> {
        let mut args = 0;
        while let Ok(token) = &self.token.0 {
            match token {
                Token::Simple(SimpleToken::ClosingParenthesis) => break,
                Token::Simple(SimpleToken::Comma) => {
                    bump!(self);
                    self = self.parse_fn_arg(&mut args)?;
                }
                _ => self = self.parse_fn_arg(&mut args)?,
            }
        }
        let span = self.context.diagnostics().merge_spans(&left, &self.token.1);
        self.context.apply_operator((Operator::FnCall(args), span));
        bump!(self);
        Ok(self)
    }

    fn parse_fn_arg(mut self, args: &mut usize) -> ParserResult<Self, Ctx, S> {
        self = self.parse_expression()?;
        *args += 1;
        Ok(self)
    }

    fn parse_atomic_expr(mut self) -> ParserResult<Self, Ctx, S> {
        match self.token.0 {
            Ok(Token::Simple(Eof)) | Ok(Token::Simple(Eol)) => {
                Err((self, ExprParsingError::NothingParsed))
            }
            Ok(Token::Ident(ident)) => {
                self.context
                    .push_atom((ExprAtom::Ident(ident), self.token.1));
                bump!(self);
                Ok(self)
            }
            Ok(Token::Literal(literal)) => {
                self.context
                    .push_atom((ExprAtom::Literal(literal), self.token.1));
                bump!(self);
                Ok(self)
            }
            Ok(Token::Simple(SimpleToken::Dot)) => {
                self.context
                    .push_atom((ExprAtom::LocationCounter, self.token.1));
                bump!(self);
                Ok(self)
            }
            _ => {
                let span = self.token.1;
                let stripped = self.context.diagnostics().strip_span(&span);
                bump!(self);
                Err((
                    self,
                    ExprParsingError::Other(
                        Message::UnexpectedToken { token: stripped }.at(span).into(),
                    ),
                ))
            }
        }
    }
}

impl<'a, Id, L, C, E, I, Ctx, S> Parser<'a, (Result<Token<Id, L, C>, E>, S), I, Ctx>
where
    I: Iterator<Item = (Result<Token<Id, L, C>, E>, S)>,
    Ctx: MacroParamsContext<S, Command = C, Ident = Id, Literal = L>,
    S: Clone,
{
    fn parse_macro_param(mut self) -> Self {
        match self.token.0 {
            Ok(Token::Ident(ident)) => self.context.add_parameter((ident, self.token.1)),
            _ => {
                let stripped = self.context.diagnostics().strip_span(&self.token.1);
                self.context.diagnostics().emit_diagnostic(
                    Message::UnexpectedToken { token: stripped }.at(self.token.1.clone()),
                )
            }
        };
        bump!(self);
        self
    }
}

impl<'a, Id, L, C, E, I, Ctx, S> Parser<'a, (Result<Token<Id, L, C>, E>, S), I, Ctx>
where
    I: Iterator<Item = (Result<Token<Id, L, C>, E>, S)>,
    Ctx: MacroInvocationContext<S, Token = Token<Id, L, C>>,
    S: Clone,
{
    fn parse_macro_arg_list(self) -> Self {
        self.parse_terminated_list(Comma.into(), LINE_FOLLOW_SET, |p| {
            let mut state = p.change_context(|c| c.enter_macro_arg());
            loop {
                match state.token {
                    (Ok(Token::Simple(Comma)), _)
                    | (Ok(Token::Simple(Eol)), _)
                    | (Ok(Token::Simple(Eof)), _) => break,
                    (Ok(other), span) => {
                        bump!(state);
                        state.context.push_token((other, span))
                    }
                    (Err(_), _) => unimplemented!(),
                }
            }
            state.change_context(|c| c.exit())
        })
    }
}

impl<'a, Id, L, C, E, I, Ctx, S> Parser<'a, (Result<Token<Id, L, C>, E>, S), I, Ctx>
where
    I: Iterator<Item = (Result<Token<Id, L, C>, E>, S)>,
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
                .emit_diagnostic(Message::UnexpectedToken { token: stripped }.at(unexpected_span));
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
        while self.token_kind() == Some(delimiter) {
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
    use crate::diag::span::{MergeSpans, StripSpan};
    use crate::diag::{CompactDiagnostic, EmitDiagnostic, Message};
    use crate::syntax::ExprAtom;
    use std::borrow::Borrow;
    use std::collections::HashMap;

    #[test]
    fn parse_empty_src() {
        assert_eq_actions(input_tokens![], [])
    }

    macro_rules! impl_diag_traits {
        ($($t:ty),* $(,)?) => {
            $(
                impl MergeSpans<SymSpan> for $t {
                    fn merge_spans(&mut self, left: &SymSpan, right: &SymSpan) -> SymSpan {
                        SymSpan::merge(left.clone(), right.clone())
                    }
                }

                impl StripSpan<SymSpan> for $t {
                    type Stripped = SymSpan;

                    fn strip_span(&mut self, span: &SymSpan) -> Self::Stripped {
                        span.clone()
                    }
                }

                impl DelegateDiagnostics<SymSpan> for $t {
                    type Delegate = Self;

                    fn diagnostics(&mut self) -> &mut Self::Delegate {
                        self
                    }
                }
            )*
        };
    }

    impl_diag_traits! {
        FileActionCollector,
        StmtActionCollector,
        CommandActionCollector,
        ArgActionCollector,
        ExprActionCollector,
        MacroParamsActionCollector,
        MacroBodyActionCollector,
        MacroInvocationActionCollector,
        MacroArgActionCollector,
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

    impl EmitDiagnostic<SymSpan, SymSpan> for FileActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<SymSpan, SymSpan>>) {
            self.actions
                .push(FileAction::EmitDiagnostic(diagnostic.into()))
        }
    }

    impl FileContext<SymIdent, SymLiteral, SymCommand, SymSpan> for FileActionCollector {
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

    impl EmitDiagnostic<SymSpan, SymSpan> for StmtActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<SymSpan, SymSpan>>) {
            self.actions
                .push(StmtAction::EmitDiagnostic(diagnostic.into()))
        }
    }

    impl StmtContext<SymIdent, SymLiteral, SymCommand, SymSpan> for StmtActionCollector {
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

    impl EmitDiagnostic<SymSpan, SymSpan> for CommandActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<SymSpan, SymSpan>>) {
            self.actions
                .push(CommandAction::EmitDiagnostic(diagnostic.into()))
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

    impl EmitDiagnostic<SymSpan, SymSpan> for ArgActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<SymSpan, SymSpan>>) {
            self.expr_action_collector
                .emit_diagnostic(diagnostic.into())
        }
    }

    impl ExprContext<SymSpan> for ArgActionCollector {
        type Ident = SymIdent;
        type Literal = SymLiteral;
        type Parent = CommandActionCollector;

        fn push_atom(&mut self, atom: (ExprAtom<SymIdent, SymLiteral>, SymSpan)) {
            self.expr_action_collector.push_atom(atom)
        }

        fn apply_operator(&mut self, operator: (Operator, SymSpan)) {
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

    impl EmitDiagnostic<SymSpan, SymSpan> for ExprActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<SymSpan, SymSpan>>) {
            self.actions
                .push(ExprAction::EmitDiagnostic(diagnostic.into()))
        }
    }

    impl ExprContext<SymSpan> for ExprActionCollector {
        type Ident = SymIdent;
        type Literal = SymLiteral;
        type Parent = Vec<ExprAction<SymSpan>>;

        fn push_atom(&mut self, atom: (ExprAtom<SymIdent, SymLiteral>, SymSpan)) {
            self.actions.push(ExprAction::PushAtom(atom))
        }

        fn apply_operator(&mut self, operator: (Operator, SymSpan)) {
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

    impl EmitDiagnostic<SymSpan, SymSpan> for MacroParamsActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<SymSpan, SymSpan>>) {
            self.actions
                .push(MacroParamsAction::EmitDiagnostic(diagnostic.into()))
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

    impl EmitDiagnostic<SymSpan, SymSpan> for MacroBodyActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<SymSpan, SymSpan>>) {
            self.actions
                .push(TokenSeqAction::EmitDiagnostic(diagnostic.into()))
        }
    }

    impl TokenSeqContext<SymSpan> for MacroBodyActionCollector {
        type Token = Token<SymIdent, SymLiteral, SymCommand>;
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

    impl EmitDiagnostic<SymSpan, SymSpan> for MacroInvocationActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<SymSpan, SymSpan>>) {
            self.actions
                .push(MacroInvocationAction::EmitDiagnostic(diagnostic.into()))
        }
    }

    impl MacroInvocationContext<SymSpan> for MacroInvocationActionCollector {
        type Token = Token<SymIdent, SymLiteral, SymCommand>;
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

    impl EmitDiagnostic<SymSpan, SymSpan> for MacroArgActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<SymSpan, SymSpan>>) {
            self.actions
                .push(TokenSeqAction::EmitDiagnostic(diagnostic.into()))
        }
    }

    impl TokenSeqContext<SymSpan> for MacroArgActionCollector {
        type Token = Token<SymIdent, SymLiteral, SymCommand>;
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
    ) -> impl Iterator<Item = (Result<SymToken, ()>, SymSpan)> {
        tokens.into_iter().cloned().map(|(t, r)| (Ok(t), r.into()))
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
            w @ Ident(()),
            plus1 @ Plus,
            x @ Ident(()),
            plus2 @ Plus,
            y @ Literal(()),
            plus3 @ Plus,
            z @ Ident(()),
        ];
        let expected = expr()
            .ident("w")
            .ident("x")
            .plus("plus1")
            .literal("y")
            .plus("plus2")
            .ident("z")
            .plus("plus3");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_subtraction() {
        let tokens = input_tokens![x @ Ident(()), minus @ Minus, y @ Literal(())];
        let expected = expr().ident("x").literal("y").minus("minus");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_division() {
        let tokens = input_tokens![x @ Ident(()), slash @ Slash, y @ Literal(())];
        let expected = expr().ident("x").literal("y").divide("slash");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn multiplication_precedes_addition() {
        let tokens = input_tokens![
            a @ Literal(()),
            plus @ Plus,
            b @ Literal(()),
            star @ Star,
            c @ Literal(()),
        ];
        let expected = expr()
            .literal("a")
            .literal("b")
            .literal("c")
            .multiply("star")
            .plus("plus");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_sum_of_terms() {
        let tokens = input_tokens![
            a @ Literal(()),
            slash @ Slash,
            b @ Literal(()),
            plus @ Plus,
            c @ Literal(()),
            star @ Star,
            d @ Ident(()),
        ];
        let expected = expr()
            .literal("a")
            .literal("b")
            .divide("slash")
            .literal("c")
            .ident("d")
            .multiply("star")
            .plus("plus");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_multiplication() {
        let tokens = input_tokens![x @ Ident(()), star @ Star, y @ Literal(())];
        let expected = expr().ident("x").literal("y").multiply("star");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_bitwise_or() {
        let tokens = input_tokens![x @ Ident(()), pipe @ Pipe, y @ Literal(())];
        let expected = expr().ident("x").literal("y").bitwise_or("pipe");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn addition_precedes_bitwise_or() {
        let tokens =
            input_tokens![x @ Ident(()), pipe @ Pipe, y @ Ident(()), plus @ Plus, z @ Ident(())];
        let expected = expr()
            .ident("x")
            .ident("y")
            .ident("z")
            .plus("plus")
            .bitwise_or("pipe");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_nullary_fn_call() {
        let tokens =
            input_tokens![name @ Ident(()), left @ OpeningParenthesis, right @ ClosingParenthesis];
        let expected = expr().ident("name").fn_call(
            0,
            SymSpan::merge(TokenRef::from("left"), TokenRef::from("right")),
        );
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_unary_fn_call() {
        let tokens = input_tokens![
            name @ Ident(()),
            left @ OpeningParenthesis,
            arg @ Ident(()),
            right @ ClosingParenthesis
        ];
        let expected = expr().ident("name").ident("arg").fn_call(
            1,
            SymSpan::merge(TokenRef::from("left"), TokenRef::from("right")),
        );
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_binary_fn_call() {
        let tokens = input_tokens![
            name @ Ident(()),
            left @ OpeningParenthesis,
            arg1 @ Ident(()),
            Simple(Comma),
            arg2 @ Ident(()),
            right @ ClosingParenthesis
        ];
        let expected = expr().ident("name").ident("arg1").ident("arg2").fn_call(
            2,
            SymSpan::merge(TokenRef::from("left"), TokenRef::from("right")),
        );
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_fn_call_plus_literal() {
        let tokens = input_tokens![
            name @ Ident(()),
            left @ OpeningParenthesis,
            right @ ClosingParenthesis,
            plus @ Simple(Plus),
            literal @ Literal(())
        ];
        let expected = expr()
            .ident("name")
            .fn_call(
                0,
                SymSpan::merge(TokenRef::from("left"), TokenRef::from("right")),
            )
            .literal("literal")
            .plus("plus");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn fn_call_precedes_multiplication() {
        let tokens = input_tokens![
            literal @ Literal(()),
            star @ Simple(Star),
            name @ Ident(()),
            left @ OpeningParenthesis,
            right @ ClosingParenthesis,
        ];
        let expected = expr()
            .literal("literal")
            .ident("name")
            .fn_call(
                0,
                SymSpan::merge(TokenRef::from("left"), TokenRef::from("right")),
            )
            .multiply("star");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_location_counter() {
        let tokens = input_tokens![dot @ Simple(Dot)];
        let expected = expr().location_counter("dot");
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
                Message::UnexpectedToken {
                    token: span.clone(),
                }
                .at(span)
                .into(),
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
            Message::UnexpectedToken {
                token: span.clone(),
            }
            .at(span)
            .into(),
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
                params: vec![MacroParamsAction::EmitDiagnostic(
                    Message::UnexpectedToken {
                        token: span.clone(),
                    }
                    .at(span)
                    .into(),
                )],
                body: vec![TokenSeqAction::PushToken((
                    Eof.into(),
                    TokenRef::from(3).into(),
                ))],
            }])],
        )
    }
}
