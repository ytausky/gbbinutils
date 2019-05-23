use super::SimpleToken::*;
use super::Token::*;
use super::{ExprAtom, Operator, Token, UnaryOperator};

use crate::analysis::syntax::*;
use crate::diag::span::{MergeSpans, StripSpan};
use crate::diag::{CompactDiagnostic, EmitDiag, Merge, Message};
use crate::model::BinOp;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::iter;

pub(super) type MockSpan = crate::diag::MockSpan<TokenRef>;

pub(super) fn with_spans<'a>(
    tokens: impl IntoIterator<Item = &'a (SymToken, TokenRef)>,
) -> impl Iterator<Item = (Result<SymToken, ()>, MockSpan)> {
    tokens.into_iter().cloned().map(|(t, r)| (Ok(t), r.into()))
}

macro_rules! impl_diag_traits {
    ($($t:ty),* $(,)?) => {
        $(
            impl MergeSpans<MockSpan> for $t {
                fn merge_spans(&mut self, left: &MockSpan, right: &MockSpan) -> MockSpan {
                    MockSpan::merge(left.clone(), right.clone())
                }
            }

            impl StripSpan<MockSpan> for $t {
                type Stripped = MockSpan;

                fn strip_span(&mut self, span: &MockSpan) -> Self::Stripped {
                    span.clone()
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

pub(super) struct FileActionCollector {
    pub actions: Vec<FileAction<MockSpan>>,
}

impl FileActionCollector {
    pub fn new() -> FileActionCollector {
        FileActionCollector {
            actions: Vec::new(),
        }
    }
}

impl EmitDiag<MockSpan, MockSpan> for FileActionCollector {
    fn emit_diag(&mut self, diagnostic: impl Into<CompactDiagnostic<MockSpan, MockSpan>>) {
        self.actions
            .push(FileAction::EmitDiagnostic(diagnostic.into()))
    }
}

impl FileContext<SymIdent, SymLiteral, SymCommand, MockSpan> for FileActionCollector {
    type StmtContext = StmtActionCollector;
    type LabelContext = LabelActionCollector;

    fn enter_unlabeled_stmt(self) -> StmtActionCollector {
        StmtActionCollector {
            label: None,
            actions: Vec::new(),
            parent: self,
        }
    }

    fn enter_labeled_stmt(self, label: (SymIdent, MockSpan)) -> Self::LabelContext {
        LabelActionCollector {
            label,
            actions: Vec::new(),
            parent: self,
        }
    }
}

pub(super) struct LabelActionCollector {
    label: (SymIdent, MockSpan),
    actions: Vec<ParamsAction<MockSpan>>,
    parent: FileActionCollector,
}

impl EmitDiag<MockSpan, MockSpan> for LabelActionCollector {
    fn emit_diag(&mut self, diagnostic: impl Into<CompactDiagnostic<MockSpan, MockSpan>>) {
        self.actions
            .push(ParamsAction::EmitDiagnostic(diagnostic.into()))
    }
}

impl ParamsContext<SymIdent, MockSpan> for LabelActionCollector {
    type Next = StmtActionCollector;

    fn add_parameter(&mut self, param: (SymIdent, MockSpan)) {
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

pub(super) struct StmtActionCollector {
    label: Option<Label<MockSpan>>,
    actions: Vec<StmtAction<MockSpan>>,
    parent: FileActionCollector,
}

impl EmitDiag<MockSpan, MockSpan> for StmtActionCollector {
    fn emit_diag(&mut self, diagnostic: impl Into<CompactDiagnostic<MockSpan, MockSpan>>) {
        self.actions
            .push(StmtAction::EmitDiagnostic(diagnostic.into()))
    }
}

impl StmtContext<SymIdent, SymLiteral, SymCommand, MockSpan> for StmtActionCollector {
    type CommandContext = CommandActionCollector;
    type MacroDefContext = MacroBodyActionCollector;
    type MacroCallContext = MacroCallActionCollector;
    type Parent = FileActionCollector;

    fn enter_command(self, command: (SymCommand, MockSpan)) -> CommandActionCollector {
        CommandActionCollector {
            command,
            actions: Vec::new(),
            parent: self,
        }
    }

    fn enter_macro_def(self, keyword: MockSpan) -> Self::MacroDefContext {
        Self::MacroDefContext {
            keyword,
            actions: Vec::new(),
            parent: self,
        }
    }

    fn enter_macro_call(self, name: (SymIdent, MockSpan)) -> MacroCallActionCollector {
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

pub(super) struct CommandActionCollector {
    command: (SymCommand, MockSpan),
    actions: Vec<CommandAction<MockSpan>>,
    parent: StmtActionCollector,
}

impl EmitDiag<MockSpan, MockSpan> for CommandActionCollector {
    fn emit_diag(&mut self, diagnostic: impl Into<CompactDiagnostic<MockSpan, MockSpan>>) {
        self.actions
            .push(CommandAction::EmitDiagnostic(diagnostic.into()))
    }
}

impl CommandContext<MockSpan> for CommandActionCollector {
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

pub(super) struct ExprActionCollector<P> {
    actions: Vec<ExprAction<MockSpan>>,
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

impl<P> EmitDiag<MockSpan, MockSpan> for ExprActionCollector<P> {
    fn emit_diag(&mut self, diagnostic: impl Into<CompactDiagnostic<MockSpan, MockSpan>>) {
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
    type ReturnTo = Vec<ExprAction<MockSpan>>;

    fn exit(self) -> Self::ReturnTo {
        self.actions
    }
}

impl<P> ExprContext<MockSpan> for ExprActionCollector<P>
where
    Self: Diagnostics<MockSpan>,
{
    type Ident = SymIdent;
    type Literal = SymLiteral;

    fn push_atom(&mut self, atom: (ExprAtom<SymIdent, SymLiteral>, MockSpan)) {
        self.actions.push(ExprAction::PushAtom(atom))
    }

    fn apply_operator(&mut self, operator: (Operator, MockSpan)) {
        self.actions.push(ExprAction::ApplyOperator(operator))
    }
}

pub(super) struct MacroBodyActionCollector {
    keyword: MockSpan,
    actions: Vec<TokenSeqAction<MockSpan>>,
    parent: StmtActionCollector,
}

impl EmitDiag<MockSpan, MockSpan> for MacroBodyActionCollector {
    fn emit_diag(&mut self, diagnostic: impl Into<CompactDiagnostic<MockSpan, MockSpan>>) {
        self.actions
            .push(TokenSeqAction::EmitDiagnostic(diagnostic.into()))
    }
}

impl TokenSeqContext<MockSpan> for MacroBodyActionCollector {
    type Token = Token<SymIdent, SymLiteral, SymCommand>;
    type Parent = StmtActionCollector;

    fn push_token(&mut self, token: (Self::Token, MockSpan)) {
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

pub(super) struct MacroCallActionCollector {
    name: (SymIdent, MockSpan),
    actions: Vec<MacroCallAction<MockSpan>>,
    parent: StmtActionCollector,
}

impl EmitDiag<MockSpan, MockSpan> for MacroCallActionCollector {
    fn emit_diag(&mut self, diagnostic: impl Into<CompactDiagnostic<MockSpan, MockSpan>>) {
        self.actions
            .push(MacroCallAction::EmitDiagnostic(diagnostic.into()))
    }
}

impl MacroCallContext<MockSpan> for MacroCallActionCollector {
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

pub(super) struct MacroArgActionCollector {
    actions: Vec<TokenSeqAction<MockSpan>>,
    parent: MacroCallActionCollector,
}

impl EmitDiag<MockSpan, MockSpan> for MacroArgActionCollector {
    fn emit_diag(&mut self, diagnostic: impl Into<CompactDiagnostic<MockSpan, MockSpan>>) {
        self.actions
            .push(TokenSeqAction::EmitDiagnostic(diagnostic.into()))
    }
}

impl TokenSeqContext<MockSpan> for MacroArgActionCollector {
    type Token = Token<SymIdent, SymLiteral, SymCommand>;
    type Parent = MacroCallActionCollector;

    fn push_token(&mut self, token: (Self::Token, MockSpan)) {
        self.actions.push(TokenSeqAction::PushToken(token))
    }

    fn exit(mut self) -> MacroCallActionCollector {
        self.parent
            .actions
            .push(MacroCallAction::MacroArg(self.actions));
        self.parent
    }
}

pub(super) fn expr() -> SymExpr {
    SymExpr(Vec::new())
}

impl SymExpr {
    pub fn ident(self, token: impl Into<TokenRef>) -> Self {
        self.push(token, |t| ExprAtom::Ident(SymIdent(t)))
    }

    pub fn literal(self, token: impl Into<TokenRef>) -> Self {
        self.push(token, |t| ExprAtom::Literal(SymLiteral(t)))
    }

    pub fn location_counter(self, token: impl Into<TokenRef>) -> Self {
        self.push(token, |_| ExprAtom::LocationCounter)
    }

    fn push(
        mut self,
        token: impl Into<TokenRef>,
        atom_ctor: impl Fn(TokenRef) -> ExprAtom<SymIdent, SymLiteral>,
    ) -> Self {
        let token_ref = token.into();
        self.0.push(ExprAction::PushAtom((
            atom_ctor(token_ref.clone()),
            token_ref.into(),
        )));
        self
    }

    pub fn divide(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator((
            Operator::Binary(BinOp::Division),
            token.into().into(),
        )));
        self
    }

    pub fn multiply(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator((
            Operator::Binary(BinOp::Multiplication),
            token.into().into(),
        )));
        self
    }

    pub fn parentheses(mut self, left: impl Into<TokenRef>, right: impl Into<TokenRef>) -> Self {
        let span = MockSpan::merge(left.into(), right.into());
        self.0.push(ExprAction::ApplyOperator((
            Operator::Unary(UnaryOperator::Parentheses),
            span,
        )));
        self
    }

    pub fn plus(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator((
            Operator::Binary(BinOp::Plus),
            token.into().into(),
        )));
        self
    }

    pub fn minus(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator((
            Operator::Binary(BinOp::Minus),
            token.into().into(),
        )));
        self
    }

    pub fn bitwise_or(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator((
            Operator::Binary(BinOp::BitwiseOr),
            token.into().into(),
        )));
        self
    }

    pub fn fn_call(mut self, args: usize, span: impl Into<MockSpan>) -> Self {
        self.0.push(ExprAction::ApplyOperator((
            Operator::FnCall(args),
            span.into(),
        )));
        self
    }

    pub fn error(mut self, message: Message<MockSpan>, highlight: impl Into<MockSpan>) -> Self {
        self.0.push(ExprAction::EmitDiagnostic(
            message.at(highlight.into()).into(),
        ));
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymCommand(pub TokenRef);

#[derive(Clone, Debug, PartialEq)]
pub struct SymIdent(pub TokenRef);

#[derive(Clone, Debug, PartialEq)]
pub struct SymLiteral(pub TokenRef);

pub type SymToken = Token<SymIdent, SymLiteral, SymCommand>;

pub fn mk_sym_token(id: impl Into<TokenRef>, token: Token<(), (), ()>) -> (SymToken, TokenRef) {
    let token_ref = id.into();
    (
        match token {
            Command(()) => Command(SymCommand(token_ref.clone())),
            Ident(()) => Ident(SymIdent(token_ref.clone())),
            Label(()) => Label(SymIdent(token_ref.clone())),
            Literal(()) => Literal(SymLiteral(token_ref.clone())),
            Simple(simple) => Simple(simple),
        },
        token_ref,
    )
}

pub(super) struct InputTokens {
    pub tokens: Vec<(SymToken, TokenRef)>,
    pub names: HashMap<String, usize>,
}

impl InputTokens {
    pub fn insert_token(&mut self, id: impl Into<TokenRef>, token: Token<(), (), ()>) {
        self.tokens.push(mk_sym_token(id, token))
    }

    pub fn token_seq<T>(&self, tokens: impl Borrow<[T]>) -> Vec<TokenSeqAction<MockSpan>>
    where
        T: Clone + Into<TokenRef>,
    {
        tokens
            .borrow()
            .iter()
            .cloned()
            .map(Into::into)
            .map(|t| TokenSeqAction::PushToken((self.token(t.clone()), t.into())))
            .collect()
    }

    fn token(&self, token_ref: impl Into<TokenRef>) -> SymToken {
        let id = match token_ref.into() {
            TokenRef::Id(n) => n,
            TokenRef::Name(name) => self.names[&name],
        };
        self.tokens[id].0.clone()
    }
}

macro_rules! add_token {
    ($input:expr, $token:expr) => {
        let id = $input.tokens.len();
        $input.insert_token(id, $token.into())
    };
    ($input:expr, $name:ident @ $token:expr) => {
        let id = stringify!($name);
        $input.names.insert(id.into(), $input.tokens.len());
        $input.insert_token(id, $token.into())
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
    ($($tokens:tt)*) => {{
        let mut input = InputTokens {
            tokens: Vec::new(),
            names: std::collections::HashMap::new(),
        };
        input_tokens_impl!(input, $($tokens)*);
        if input
            .tokens
            .last()
            .map(|(token, _)| *token != Eof.into())
            .unwrap_or(true)
        {
            let eof_id = input.tokens.len().into();
            input.tokens.push((Eof.into(), eof_id))
        }
        input
    }};
}

#[derive(Clone, Debug, PartialEq)]
pub enum TokenRef {
    Id(usize),
    Name(String),
}

impl From<usize> for TokenRef {
    fn from(id: usize) -> Self {
        TokenRef::Id(id)
    }
}

impl From<&'static str> for TokenRef {
    fn from(name: &'static str) -> Self {
        TokenRef::Name(name.to_string())
    }
}

#[derive(Debug, PartialEq)]
pub(super) enum FileAction<S> {
    Stmt {
        label: Option<Label<S>>,
        actions: Vec<StmtAction<S>>,
    },
    EmitDiagnostic(CompactDiagnostic<S, S>),
}

pub(super) type Label<S> = ((SymIdent, S), Vec<ParamsAction<S>>);

#[derive(Debug, PartialEq)]
pub(super) enum StmtAction<S> {
    Command {
        command: (SymCommand, S),
        actions: Vec<CommandAction<S>>,
    },
    MacroDef {
        keyword: S,
        body: Vec<TokenSeqAction<S>>,
    },
    MacroCall {
        name: (SymIdent, S),
        actions: Vec<MacroCallAction<S>>,
    },
    EmitDiagnostic(CompactDiagnostic<S, S>),
}

#[derive(Debug, PartialEq)]
pub(super) enum CommandAction<S> {
    AddArgument { actions: Vec<ExprAction<S>> },
    EmitDiagnostic(CompactDiagnostic<S, S>),
}

#[derive(Debug, PartialEq, Clone)]
pub(super) enum ExprAction<S> {
    PushAtom((ExprAtom<SymIdent, SymLiteral>, S)),
    ApplyOperator((Operator, S)),
    EmitDiagnostic(CompactDiagnostic<S, S>),
}

#[derive(Debug, PartialEq)]
pub(super) enum ParamsAction<S> {
    AddParameter((SymIdent, S)),
    EmitDiagnostic(CompactDiagnostic<S, S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum TokenSeqAction<S> {
    PushToken((Token<SymIdent, SymLiteral, SymCommand>, S)),
    EmitDiagnostic(CompactDiagnostic<S, S>),
}

#[derive(Debug, PartialEq)]
pub(super) enum MacroCallAction<S> {
    MacroArg(Vec<TokenSeqAction<S>>),
    EmitDiagnostic(CompactDiagnostic<S, S>),
}

#[derive(Clone)]
pub(super) struct SymExpr(pub Vec<ExprAction<MockSpan>>);

pub(super) fn labeled(
    label: impl Into<TokenRef>,
    params: impl Borrow<[TokenRef]>,
    actions: Vec<StmtAction<MockSpan>>,
) -> FileAction<MockSpan> {
    let label = label.into();
    FileAction::Stmt {
        label: Some((
            (SymIdent(label.clone()), label.into()),
            convert_params(params),
        )),
        actions,
    }
}

pub(super) fn unlabeled(actions: Vec<StmtAction<MockSpan>>) -> FileAction<MockSpan> {
    FileAction::Stmt {
        label: None,
        actions,
    }
}

pub(super) fn empty() -> Vec<StmtAction<MockSpan>> {
    Vec::new()
}

pub(super) fn command(
    id: impl Into<TokenRef>,
    args: impl Borrow<[SymExpr]>,
) -> Vec<StmtAction<MockSpan>> {
    let id = id.into();
    vec![StmtAction::Command {
        command: (SymCommand(id.clone()), id.into()),
        actions: args
            .borrow()
            .iter()
            .cloned()
            .map(|SymExpr(expr)| CommandAction::AddArgument { actions: expr })
            .collect(),
    }]
}

pub(super) fn malformed_command(
    id: impl Into<TokenRef>,
    args: impl Borrow<[SymExpr]>,
    diagnostic: CompactDiagnostic<MockSpan, MockSpan>,
) -> Vec<StmtAction<MockSpan>> {
    let id = id.into();
    vec![StmtAction::Command {
        command: (SymCommand(id.clone()), id.into()),
        actions: args
            .borrow()
            .iter()
            .cloned()
            .map(|SymExpr(expr)| CommandAction::AddArgument { actions: expr })
            .chain(iter::once(CommandAction::EmitDiagnostic(diagnostic)))
            .collect(),
    }]
}

pub(super) fn call_macro(
    id: impl Into<TokenRef>,
    args: impl Borrow<[Vec<TokenSeqAction<MockSpan>>]>,
) -> Vec<StmtAction<MockSpan>> {
    let id = id.into();
    vec![StmtAction::MacroCall {
        name: (SymIdent(id.clone()), id.into()),
        actions: args
            .borrow()
            .iter()
            .cloned()
            .map(MacroCallAction::MacroArg)
            .collect(),
    }]
}

pub(super) fn macro_def(
    keyword: impl Into<TokenRef>,
    mut body: Vec<TokenSeqAction<MockSpan>>,
    endm: impl Into<TokenRef>,
) -> Vec<StmtAction<MockSpan>> {
    body.push(TokenSeqAction::PushToken((Eof.into(), endm.into().into())));
    vec![StmtAction::MacroDef {
        keyword: keyword.into().into(),
        body,
    }]
}

fn convert_params(params: impl Borrow<[TokenRef]>) -> Vec<ParamsAction<MockSpan>> {
    params
        .borrow()
        .iter()
        .cloned()
        .map(|t| ParamsAction::AddParameter((SymIdent(t.clone()), t.into())))
        .collect()
}

pub(super) fn push_token(
    token: impl Into<SymToken>,
    span: impl Into<TokenRef>,
) -> TokenSeqAction<MockSpan> {
    TokenSeqAction::PushToken((token.into(), span.into().into()))
}

pub(super) fn malformed_macro_def(
    keyword: impl Into<TokenRef>,
    mut body: Vec<TokenSeqAction<MockSpan>>,
    diagnostic: CompactDiagnostic<MockSpan, MockSpan>,
) -> Vec<StmtAction<MockSpan>> {
    body.push(TokenSeqAction::EmitDiagnostic(diagnostic));
    vec![StmtAction::MacroDef {
        keyword: keyword.into().into(),
        body,
    }]
}

pub(super) fn stmt_error(
    message: Message<MockSpan>,
    highlight: impl Into<TokenRef>,
) -> Vec<StmtAction<MockSpan>> {
    vec![StmtAction::EmitDiagnostic(arg_error(message, highlight))]
}

pub(super) fn arg_error(
    message: Message<MockSpan>,
    highlight: impl Into<TokenRef>,
) -> CompactDiagnostic<MockSpan, MockSpan> {
    message.at(highlight.into().into()).into()
}

mod tests {
    use super::*;

    #[test]
    fn test_token_macro() {
        let tokens = input_tokens![
            my_tok @ Command(()),
            Literal(()),
            next_one @ Macro,
        ];
        assert_eq!(
            tokens.tokens,
            [
                (Command(SymCommand("my_tok".into())), "my_tok".into()),
                (Literal(SymLiteral(1.into())), 1.into()),
                (Macro.into(), "next_one".into()),
                (Eof.into(), 3.into()),
            ]
        );
        assert_eq!(tokens.names.get("my_tok"), Some(&0));
        assert_eq!(tokens.names.get("next_one"), Some(&2))
    }
}
