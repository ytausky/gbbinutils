use super::SimpleToken::*;
use super::*;
use crate::diag::span::StripSpan;
use crate::diag::{CompactDiagnostic, EmitDiagnostic, Message};

macro_rules! bump {
    ($parser:expr) => {
        $parser.token = $parser.remaining.next().unwrap()
    };
}

mod expr;

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
    recovery: Option<RecoveryState>,
    context: C,
}

enum RecoveryState {
    DiagnosedEof,
}

impl<'a, T, I: Iterator<Item = T>, C> Parser<'a, T, I, C> {
    fn new(tokens: &'a mut I, context: C) -> Self {
        Parser {
            token: tokens.next().unwrap(),
            remaining: tokens,
            recovery: None,
            context,
        }
    }
}

impl<'a, T, I, C> Parser<'a, T, I, C> {
    fn change_context<D, F: FnOnce(C) -> D>(self, f: F) -> Parser<'a, T, I, D> {
        Parser {
            token: self.token,
            remaining: self.remaining,
            recovery: self.recovery,
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
        if let (Ok(Token::Label(label)), span) = self.token {
            bump!(self);
            let mut parser = self.change_context(|c| c.enter_labeled_stmt((label, span)));
            if parser.token_kind() == Some(LParen.into()) {
                bump!(parser);
                parser = parser.parse_terminated_list(
                    Comma.into(),
                    &[RParen.into()],
                    Parser::parse_param,
                );
                if parser.token_kind() == Some(RParen.into()) {
                    bump!(parser)
                } else {
                    parser = parser.diagnose_unexpected_token();
                    return parser
                        .change_context(ParamsContext::next)
                        .change_context(StmtContext::exit);
                }
            }
            parser.change_context(ParamsContext::next)
        } else {
            self.change_context(FileContext::enter_unlabeled_stmt)
        }
        .parse_unlabeled_stmt()
        .change_context(StmtContext::exit)
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
                    self.parse_macro_call((ident, span))
                }
                (Ok(Token::Simple(SimpleToken::Macro)), span) => {
                    bump!(self);
                    self.parse_macro_def(span)
                }
                (_, span) => {
                    bump!(self);
                    let stripped = self.context.diagnostics().strip_span(&span);
                    self.emit_diagnostic(Message::UnexpectedToken { token: stripped }.at(span));
                    self
                }
            };
        }
    }

    fn parse_command(self, command: (C, S)) -> Self {
        self.change_context(|c| c.enter_command(command))
            .parse_argument_list()
            .change_context(CommandContext::exit)
    }

    fn parse_macro_def(self, span: S) -> Self {
        let mut state = self.change_context(|c| c.enter_macro_def(span));
        if !state.token_is_in(LINE_FOLLOW_SET) {
            state = state.diagnose_unexpected_token();
            while !state.token_is_in(LINE_FOLLOW_SET) {
                bump!(state)
            }
        }
        if state.token_kind() == Some(Eol.into()) {
            bump!(state);
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
                        state = state.diagnose_unexpected_token();
                        break;
                    }
                    (Ok(other), span) => {
                        state.context.push_token((other, span));
                        bump!(state);
                    }
                    (Err(_), _) => unimplemented!(),
                }
            }
        } else {
            assert_eq!(state.token_kind(), Some(Eof.into()));
            state = state.diagnose_unexpected_token();
        }
        state.change_context(TokenSeqContext::exit)
    }

    fn parse_macro_call(self, name: (Id, S)) -> Self {
        self.change_context(|c| c.enter_macro_call(name))
            .parse_macro_arg_list()
            .change_context(MacroCallContext::exit)
    }
}

impl<'a, Id, L, C, E, I, Ctx, S> Parser<'a, (Result<Token<Id, L, C>, E>, S), I, Ctx>
where
    I: Iterator<Item = (Result<Token<Id, L, C>, E>, S)>,
    Ctx: CommandContext<S, Command = C, Ident = Id, Literal = L>,
    S: Clone,
{
    fn parse_argument_list(self) -> Self {
        self.parse_terminated_list(Comma.into(), LINE_FOLLOW_SET, Parser::parse_argument)
    }

    fn parse_argument(self) -> Self {
        self.change_context(CommandContext::add_argument)
            .parse()
            .change_context(FinalContext::exit)
    }
}

impl<'a, Id, L, C, E, I, Ctx, S> Parser<'a, (Result<Token<Id, L, C>, E>, S), I, Ctx>
where
    I: Iterator<Item = (Result<Token<Id, L, C>, E>, S)>,
    Ctx: ParamsContext<Id, S>,
    S: Clone,
{
    fn parse_param(mut self) -> Self {
        match self.token.0 {
            Ok(Token::Ident(ident)) => {
                self.context.add_parameter((ident, self.token.1));
                bump!(self)
            }
            _ => self = self.diagnose_unexpected_token(),
        };
        self
    }
}

impl<'a, Id, L, C, E, I, Ctx, S> Parser<'a, (Result<Token<Id, L, C>, E>, S), I, Ctx>
where
    I: Iterator<Item = (Result<Token<Id, L, C>, E>, S)>,
    Ctx: MacroCallContext<S, Token = Token<Id, L, C>>,
    S: Clone,
{
    fn parse_macro_arg_list(self) -> Self {
        self.parse_terminated_list(Comma.into(), LINE_FOLLOW_SET, |p| {
            let mut state = p.change_context(MacroCallContext::enter_macro_arg);
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
            state.change_context(TokenSeqContext::exit)
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
            self = self.diagnose_unexpected_token();
            while !self.token_is_in(terminators) && self.token_kind() != Some(Token::Simple(Eof)) {
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

    fn diagnose_unexpected_token(mut self) -> Self {
        if self.token_kind() == Some(Token::Simple(Eof)) {
            if self.recovery.is_none() {
                self.emit_diagnostic(Message::UnexpectedEof.at(self.token.1.clone()));
                self.recovery = Some(RecoveryState::DiagnosedEof)
            }
        } else {
            let token = self.token.1;
            bump!(self);
            let stripped = self.context.diagnostics().strip_span(&token);
            self.emit_diagnostic(Message::UnexpectedToken { token: stripped }.at(token))
        }
        self
    }
}

impl<'a, Id, L, C, E, I, Ctx, S> EmitDiagnostic<S, <Ctx::Delegate as StripSpan<S>>::Stripped>
    for Parser<'a, (Result<Token<Id, L, C>, E>, S), I, Ctx>
where
    I: Iterator<Item = (Result<Token<Id, L, C>, E>, S)>,
    Ctx: DelegateDiagnostics<S>,
    S: Clone,
{
    fn emit_diagnostic(
        &mut self,
        diagnostic: impl Into<CompactDiagnostic<S, <Ctx::Delegate as StripSpan<S>>::Stripped>>,
    ) {
        self.context.diagnostics().emit_diagnostic(diagnostic)
    }
}

#[cfg(test)]
pub(self) mod tests {
    use super::ast::*;
    use super::Token::*;
    use super::*;
    use crate::diag::span::{MergeSpans, StripSpan};
    use crate::diag::{CompactDiagnostic, EmitDiagnostic, Message};
    use crate::syntax::ExprAtom;
    use std::borrow::Borrow;

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
        LabelActionCollector,
        StmtActionCollector,
        CommandActionCollector,
        ExprActionCollector<CommandActionCollector>,
        ExprActionCollector<()>,
        MacroBodyActionCollector,
        MacroCallActionCollector,
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
        type LabelContext = LabelActionCollector;

        fn enter_unlabeled_stmt(self) -> StmtActionCollector {
            StmtActionCollector {
                label: None,
                actions: Vec::new(),
                parent: self,
            }
        }

        fn enter_labeled_stmt(self, label: (SymIdent, SymSpan)) -> Self::LabelContext {
            LabelActionCollector {
                label,
                actions: Vec::new(),
                parent: self,
            }
        }
    }

    struct LabelActionCollector {
        label: (SymIdent, SymSpan),
        actions: Vec<ParamsAction<SymSpan>>,
        parent: FileActionCollector,
    }

    impl EmitDiagnostic<SymSpan, SymSpan> for LabelActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<SymSpan, SymSpan>>) {
            self.actions
                .push(ParamsAction::EmitDiagnostic(diagnostic.into()))
        }
    }

    impl ParamsContext<SymIdent, SymSpan> for LabelActionCollector {
        type Next = StmtActionCollector;

        fn add_parameter(&mut self, param: (SymIdent, SymSpan)) {
            self.actions.push(ParamsAction::AddParameter(param))
        }

        fn next(self) -> Self::Next {
            Self::Next {
                label: Some((self.label, self.actions)),
                actions: Vec::new(),
                parent: self.parent,
            }
        }
    }

    struct StmtActionCollector {
        label: Option<ast::Label<SymSpan>>,
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
        type MacroDefContext = MacroBodyActionCollector;
        type MacroCallContext = MacroCallActionCollector;
        type Parent = FileActionCollector;

        fn enter_command(self, command: (SymCommand, SymSpan)) -> CommandActionCollector {
            CommandActionCollector {
                command,
                actions: Vec::new(),
                parent: self,
            }
        }

        fn enter_macro_def(self, keyword: SymSpan) -> Self::MacroDefContext {
            Self::MacroDefContext {
                keyword,
                actions: Vec::new(),
                parent: self,
            }
        }

        fn enter_macro_call(self, name: (SymIdent, SymSpan)) -> MacroCallActionCollector {
            MacroCallActionCollector {
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
        type ArgContext = ExprActionCollector<Self>;
        type Parent = StmtActionCollector;

        fn add_argument(self) -> Self::ArgContext {
            ExprActionCollector::new(self)
        }

        fn exit(mut self) -> StmtActionCollector {
            self.parent.actions.push(StmtAction::Command {
                command: self.command,
                actions: self.actions,
            });
            self.parent
        }
    }

    pub struct ExprActionCollector<P> {
        actions: Vec<ExprAction<SymSpan>>,
        parent: P,
    }

    impl<P> ExprActionCollector<P> {
        pub fn new(parent: P) -> Self {
            Self {
                actions: Vec::new(),
                parent,
            }
        }
    }

    impl<P> EmitDiagnostic<SymSpan, SymSpan> for ExprActionCollector<P> {
        fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<SymSpan, SymSpan>>) {
            self.actions
                .push(ExprAction::EmitDiagnostic(diagnostic.into()))
        }
    }

    impl FinalContext for ExprActionCollector<CommandActionCollector> {
        type ReturnTo = CommandActionCollector;

        fn exit(mut self) -> Self::ReturnTo {
            self.parent.actions.push(CommandAction::AddArgument {
                actions: self.actions,
            });
            self.parent
        }
    }

    impl FinalContext for ExprActionCollector<()> {
        type ReturnTo = Vec<ExprAction<SymSpan>>;

        fn exit(self) -> Self::ReturnTo {
            self.actions
        }
    }

    impl<P> ExprContext<SymSpan> for ExprActionCollector<P>
    where
        Self: DelegateDiagnostics<SymSpan>,
    {
        type Ident = SymIdent;
        type Literal = SymLiteral;

        fn push_atom(&mut self, atom: (ExprAtom<SymIdent, SymLiteral>, SymSpan)) {
            self.actions.push(ExprAction::PushAtom(atom))
        }

        fn apply_operator(&mut self, operator: (Operator, SymSpan)) {
            self.actions.push(ExprAction::ApplyOperator(operator))
        }
    }

    struct MacroBodyActionCollector {
        keyword: SymSpan,
        actions: Vec<TokenSeqAction<SymSpan>>,
        parent: StmtActionCollector,
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
            self.parent.actions.push(StmtAction::MacroDef {
                keyword: self.keyword,
                body: self.actions,
            });
            self.parent
        }
    }

    struct MacroCallActionCollector {
        name: (SymIdent, SymSpan),
        actions: Vec<MacroCallAction<SymSpan>>,
        parent: StmtActionCollector,
    }

    impl EmitDiagnostic<SymSpan, SymSpan> for MacroCallActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<SymSpan, SymSpan>>) {
            self.actions
                .push(MacroCallAction::EmitDiagnostic(diagnostic.into()))
        }
    }

    impl MacroCallContext<SymSpan> for MacroCallActionCollector {
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
            self.parent.actions.push(StmtAction::MacroCall {
                name: self.name,
                actions: self.actions,
            });
            self.parent
        }
    }

    struct MacroArgActionCollector {
        actions: Vec<TokenSeqAction<SymSpan>>,
        parent: MacroCallActionCollector,
    }

    impl EmitDiagnostic<SymSpan, SymSpan> for MacroArgActionCollector {
        fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<SymSpan, SymSpan>>) {
            self.actions
                .push(TokenSeqAction::EmitDiagnostic(diagnostic.into()))
        }
    }

    impl TokenSeqContext<SymSpan> for MacroArgActionCollector {
        type Token = Token<SymIdent, SymLiteral, SymCommand>;
        type Parent = MacroCallActionCollector;

        fn push_token(&mut self, token: (Self::Token, SymSpan)) {
            self.actions.push(TokenSeqAction::PushToken(token))
        }

        fn exit(mut self) -> MacroCallActionCollector {
            self.parent
                .actions
                .push(MacroCallAction::MacroArg(self.actions));
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

    pub fn with_spans<'a>(
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
        let expected_actions = [labeled(0, vec![], macro_def(1, Vec::new(), 3))];
        assert_eq_actions(tokens, expected_actions);
    }

    #[test]
    fn parse_macro_definition_with_instruction() {
        let tokens = input_tokens![Label(()), Macro, Eol, Command(()), Eol, Endm];
        let expected_actions = [labeled(
            0,
            vec![],
            macro_def(1, tokens.token_seq([3, 4]), 5),
        )];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_nonempty_macro_def_with_two_params() {
        let tokens = input_tokens![
            l @ Label(()),
            LParen,
            p1 @ Ident(()),
            Comma,
            p2 @ Ident(()),
            RParen,
            key @ Macro,
            Eol,
            t1 @ Command(()),
            t2 @ Eol,
            endm @ Endm,
        ];
        let expected = [labeled(
            "l",
            ["p1".into(), "p2".into()],
            macro_def("key", tokens.token_seq(["t1", "t2"]), "endm"),
        )];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_label() {
        let tokens = input_tokens![Label(()), Eol];
        let expected_actions = [labeled(0, vec![], empty())];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_two_consecutive_labels() {
        let tokens = input_tokens![Label(()), Eol, Label(())];
        let expected = [labeled(0, vec![], empty()), labeled(2, vec![], empty())];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_labeled_instruction() {
        let tokens = input_tokens![Label(()), Command(()), Eol];
        let expected = [labeled(0, vec![], command(1, [])), unlabeled(empty())];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_labeled_command_with_eol_separators() {
        let tokens = input_tokens![Label(()), Eol, Eol, Command(())];
        let expected = [labeled(0, vec![], command(3, []))];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_deref_operand() {
        let tokens = input_tokens![
            jp @ Command(()),
            open @ LParen,
            hl @ Literal(()),
            close @ RParen,
        ];
        let expected = [unlabeled(command(
            "jp",
            [expr().literal("hl").parentheses("open", "close")],
        ))];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_nullary_macro_call() {
        let tokens = input_tokens![Ident(())];
        let expected_actions = [unlabeled(call_macro(0, []))];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_call() {
        let tokens = input_tokens![Ident(()), Literal(())];
        let expected_actions = [unlabeled(call_macro(0, [tokens.token_seq([1])]))];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_call_with_multiple_terminals() {
        let tokens = input_tokens![Ident(()), Literal(()), Literal(()), Literal(())];
        let expected_actions = [unlabeled(call_macro(0, [tokens.token_seq([1, 2, 3])]))];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_binary_macro_call_with_multiple_terminals() {
        let tokens = input_tokens![
            Ident(()),
            Literal(()),
            Literal(()),
            Comma,
            Literal(()),
            Literal(()),
            Literal(()),
        ];
        let expected_actions = [unlabeled(call_macro(
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
    fn diagnose_eof_in_param_list() {
        assert_eq_actions(
            input_tokens![label @ Label(()), LParen, eof @ Eof],
            [FileAction::Stmt {
                label: Some((
                    (SymIdent("label".into()), TokenRef::from("label").into()),
                    vec![ParamsAction::EmitDiagnostic(arg_error(
                        Message::UnexpectedEof,
                        "eof",
                    ))],
                )),
                actions: vec![],
            }],
        )
    }

    #[test]
    fn diagnose_eof_in_macro_body() {
        assert_eq_actions(
            input_tokens![Macro, Eol, eof @ Eof],
            [unlabeled(malformed_macro_def(
                0,
                Vec::new(),
                arg_error(Message::UnexpectedEof, "eof"),
            ))],
        )
    }

    #[test]
    fn diagnose_unmatched_parentheses() {
        assert_eq_actions(
            input_tokens![Command(()), paren @ LParen, Literal(())],
            [unlabeled(command(
                0,
                [expr()
                    .literal(2)
                    .error(Message::UnmatchedParenthesis, TokenRef::from("paren"))],
            ))],
        )
    }

    #[test]
    fn recover_from_unexpected_token_in_expr() {
        let paren_span: SymSpan = TokenRef::from("paren").into();
        assert_eq_actions(
            input_tokens![
                Command(()),
                paren @ RParen,
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
            input_tokens![Command(()), LParen, Eol],
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
    fn diagnose_unexpected_token_param_list() {
        let span: SymSpan = TokenRef::from("lit").into();
        assert_eq_actions(
            input_tokens![
                label @ Label(()),
                LParen,
                lit @ Literal(()),
                RParen,
                key @ Macro,
                Eol,
                endm @ Endm
            ],
            [FileAction::Stmt {
                label: Some((
                    (SymIdent("label".into()), TokenRef::from("label").into()),
                    vec![ParamsAction::EmitDiagnostic(
                        Message::UnexpectedToken {
                            token: span.clone(),
                        }
                        .at(span)
                        .into(),
                    )],
                )),
                actions: vec![StmtAction::MacroDef {
                    keyword: TokenRef::from("key").into(),
                    body: vec![TokenSeqAction::PushToken((
                        Eof.into(),
                        TokenRef::from("endm").into(),
                    ))],
                }],
            }],
        )
    }

    #[test]
    fn diagnose_unexpected_token_after_macro_keyword() {
        let tokens = input_tokens![
            label @ Label(()),
            key @ Macro,
            unexpected @ Ident(()),
            Eol,
            t1 @ Command(()),
            t2 @ Eol,
            t3 @ Endm,
        ];
        let unexpected = TokenRef::from("unexpected");
        let mut body = vec![TokenSeqAction::EmitDiagnostic(arg_error(
            Message::UnexpectedToken {
                token: unexpected.clone().into(),
            },
            unexpected,
        ))];
        body.extend(tokens.token_seq(["t1", "t2"]));
        body.push(push_token(Eof, "t3"));
        let expected = [labeled(
            "label",
            vec![],
            vec![StmtAction::MacroDef {
                keyword: TokenRef::from("key").into(),
                body,
            }],
        )];
        assert_eq_actions(tokens, expected)
    }
}
