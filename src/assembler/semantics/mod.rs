use super::keywords::{BuiltinMnemonic, Directive, Mnemonic, OperandSymbol};
use super::session::*;
use super::syntax::actions::*;
use super::syntax::{LexError, Literal, SemanticToken, Sigil, Token};

use crate::diagnostics::{CompactDiag, Diagnostics, Message};
use crate::expr::{Atom, Expr, ExprOp, ParamId};
use crate::object::SymbolId;
use crate::span::{SpanSource, Spanned, StripSpan, WithSpan};

macro_rules! set_state {
    ($session:expr, $state:expr) => {
        $crate::assembler::semantics::Semantics {
            session: $session.session,
            state: $state,
        }
    };
}

mod cpu_instr;
mod directive;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Keyword {
    BuiltinMnemonic(BuiltinMnemonic),
    Operand(OperandSymbol),
}

pub(super) struct Semantics<'a, S, T> {
    pub session: &'a mut S,
    pub state: T,
}

impl<'a, 'b, S: Analysis, T> Semantics<'a, S, T> {
    fn map_state<F: FnOnce(T) -> U, U>(self, f: F) -> Semantics<'a, S, U> {
        Semantics {
            session: self.session,
            state: f(self.state),
        }
    }
}

type TokenStreamSemantics<'a, S> =
    Semantics<'a, S, TokenStreamState<<S as StringSource>::StringRef, <S as SpanSource>::Span>>;

#[derive(Debug, PartialEq)]
pub(super) struct TokenStreamState<R, S> {
    mode: LineRule<InstrLineState<R, S>, TokenLineState<R, S>>,
}

impl<R, S> TokenStreamState<R, S> {
    pub fn new() -> Self {
        Self {
            mode: LineRule::InstrLine(InstrLineState::new()),
        }
    }
}

type InstrLineSemantics<'a, S> =
    Semantics<'a, S, InstrLineState<<S as StringSource>::StringRef, <S as SpanSource>::Span>>;

#[derive(Debug, PartialEq)]
pub struct InstrLineState<R, S> {
    label: Option<Label<R, S>>,
}

impl<R, S> InstrLineState<R, S> {
    fn new() -> Self {
        Self { label: None }
    }
}

type Label<R, S> = ((R, S), Params<R, S>);
type Params<I, S> = (Vec<I>, Vec<S>);

type TokenLineSemantics<'a, S> =
    Semantics<'a, S, TokenLineState<<S as StringSource>::StringRef, <S as SpanSource>::Span>>;

#[derive(Debug, PartialEq)]
pub struct TokenLineState<R, S> {
    context: TokenContext<R, S>,
}

#[derive(Debug, PartialEq)]
pub enum TokenContext<R, S> {
    FalseIf,
    MacroDef(MacroDefState<R, S>),
}

#[derive(Debug, PartialEq)]
pub struct MacroDefState<R, S> {
    label: Option<Label<R, S>>,
    tokens: TokenSeq<R, S>,
}

pub type TokenSeq<R, S> = (Vec<SemanticToken<R>>, Vec<S>);

impl<R, S> MacroDefState<R, S> {
    fn new(label: Option<Label<R, S>>) -> Self {
        Self {
            label,
            tokens: (Vec::new(), Vec::new()),
        }
    }
}

type BuiltinInstrSemantics<'a, S> = Semantics<'a, S, BuiltinInstrState<S>>;

pub(super) struct BuiltinInstrState<S: Analysis> {
    label: Option<Label<S::StringRef, S::Span>>,
    mnemonic: Spanned<BuiltinMnemonic, S::Span>,
    args: BuiltinInstrArgs<S::StringRef, S::Span>,
}

impl<S: Analysis> BuiltinInstrState<S> {
    fn new(
        label: Option<Label<S::StringRef, S::Span>>,
        mnemonic: Spanned<BuiltinMnemonic, S::Span>,
    ) -> Self {
        Self {
            label,
            mnemonic,
            args: Vec::new(),
        }
    }
}

type BuiltinInstrArgs<R, S> = Vec<ParsedArg<R, S>>;

pub(super) type ArgSemantics<'a, S> = Semantics<
    'a,
    S,
    ExprBuilder<<S as StringSource>::StringRef, <S as SpanSource>::Span, BuiltinInstrState<S>>,
>;

pub(crate) struct ExprBuilder<R, S, P> {
    arg: Option<ParsedArg<R, S>>,
    parent: P,
}

impl<R, S, P> ExprBuilder<R, S, P> {
    pub fn new(parent: P) -> Self {
        Self { arg: None, parent }
    }
}

enum ParsedArg<R, S> {
    Bare(Expr<R, S>),
    Parenthesized(Expr<R, S>, S),
    String(R, S),
    Error,
}

enum Arg<R, S> {
    Bare(BareArg<S>),
    Deref(BareArg<S>, S),
    String(R, S),
    Error,
}

#[derive(Clone)]
enum BareArg<S> {
    Const(Expr<SymbolId, S>),
    Symbol(OperandSymbol, S),
}

trait NameVisibility<R> {
    fn name_visibility(&self, name: &R) -> Visibility;
}

impl<S> NameVisibility<S::StringRef> for S
where
    S: Interner + NameTable<<S as StringSource>::StringRef>,
{
    fn name_visibility(&self, name: &S::StringRef) -> Visibility {
        if self.get_string(name).starts_with('_') {
            Visibility::Local
        } else {
            Visibility::Global
        }
    }
}

trait DefineName<R> {
    fn define_name(&mut self, name: R, entry: ResolvedName);
}

impl<S> DefineName<S::StringRef> for S
where
    S: Interner + NameTable<<S as StringSource>::StringRef>,
{
    fn define_name(&mut self, name: S::StringRef, entry: ResolvedName) {
        let visibility = self.name_visibility(&name);
        self.define_name_with_visibility(name, visibility, entry)
    }
}

trait ResolveName<R> {
    fn resolve_name(&mut self, name: &R) -> Option<ResolvedName>;
}

impl<S> ResolveName<S::StringRef> for S
where
    S: Interner + NameTable<<S as StringSource>::StringRef>,
{
    fn resolve_name(&mut self, name: &S::StringRef) -> Option<ResolvedName> {
        let visibility = self.name_visibility(name);
        self.resolve_name_with_visibility(name, visibility)
    }
}

impl<R, S> From<InstrLineState<R, S>> for TokenStreamState<R, S> {
    fn from(actions: InstrLineState<R, S>) -> Self {
        Self {
            mode: LineRule::InstrLine(actions),
        }
    }
}

impl<R, S> From<TokenLineState<R, S>> for TokenStreamState<R, S> {
    fn from(actions: TokenLineState<R, S>) -> Self {
        Self {
            mode: LineRule::TokenLine(actions),
        }
    }
}

impl<'a, S, T> ParsingContext for Semantics<'a, S, T>
where
    S: NextToken + Diagnostics<<S as SpanSource>::Span>,
{
    type Ident = S::StringRef;
    type Literal = Literal<S::StringRef>;
    type Error = LexError;
    type Span = S::Span;
    type Stripped = <S as StripSpan<<S as SpanSource>::Span>>::Stripped;

    fn next_token(
        &mut self,
    ) -> Option<LexerOutput<Self::Ident, Self::Literal, Self::Error, Self::Span>> {
        self.session.next_token()
    }

    fn merge_spans(&mut self, left: &Self::Span, right: &Self::Span) -> Self::Span {
        self.session.merge_spans(left, right)
    }

    fn strip_span(&mut self, span: &Self::Span) -> Self::Stripped {
        self.session.strip_span(span)
    }

    fn emit_diag(&mut self, diag: impl Into<CompactDiag<Self::Span, Self::Stripped>>) {
        self.session.emit_diag(diag)
    }
}

impl<'a, S: Analysis> TokenStreamContext for TokenStreamSemantics<'a, S>
where
    S::StringRef: 'static,
    S::Span: 'static,
{
    type InstrLineContext = InstrLineSemantics<'a, S>;
    type TokenLineContext = TokenLineSemantics<'a, S>;
    type TokenLineFinalizer = TokenContextFinalizationSemantics<'a, S>;

    fn will_parse_line(self) -> LineRule<Self::InstrLineContext, Self::TokenLineContext> {
        match self.state.mode {
            LineRule::InstrLine(state) => LineRule::InstrLine(set_state!(self, state)),
            LineRule::TokenLine(state) => LineRule::TokenLine(set_state!(self, state)),
        }
    }

    fn act_on_eos(self, span: S::Span) -> Self {
        match self.state.mode {
            LineRule::InstrLine(state) => {
                let mut semantics = set_state!(self, state);
                semantics.flush_label();
                set_state!(semantics, semantics.state.into())
            }
            LineRule::TokenLine(ref state) => {
                match state.context {
                    TokenContext::FalseIf => unimplemented!(),
                    TokenContext::MacroDef(_) => {
                        self.session.emit_diag(Message::UnexpectedEof.at(span))
                    }
                }
                self
            }
        }
    }
}

impl<'a, S: Analysis> InstrFinalizer for InstrLineSemantics<'a, S> {
    type Next = TokenStreamSemantics<'a, S>;

    fn did_parse_instr(self) -> Self::Next {
        set_state!(self, self.state.into())
    }
}

impl<'a, S: Analysis> LineFinalizer for InstrLineSemantics<'a, S> {
    type Next = TokenStreamSemantics<'a, S>;

    fn did_parse_line(self, _: S::Span) -> Self::Next {
        set_state!(self, self.state.into())
    }
}

impl<'a, S: Analysis> LineFinalizer for TokenStreamSemantics<'a, S> {
    type Next = Self;

    fn did_parse_line(self, _: S::Span) -> Self::Next {
        self
    }
}

impl<'a, S: Analysis> TokenLineContext for TokenLineSemantics<'a, S> {
    type ContextFinalizer = TokenContextFinalizationSemantics<'a, S>;

    fn act_on_token(&mut self, token: SemanticToken<S::StringRef>, span: S::Span) {
        match &mut self.state.context {
            TokenContext::FalseIf => (),
            TokenContext::MacroDef(state) => state.act_on_token(token, span),
        }
    }

    fn act_on_mnemonic(
        mut self,
        ident: S::StringRef,
        span: S::Span,
    ) -> TokenLineRule<Self, Self::ContextFinalizer> {
        if let Some(MnemonicEntry::Builtin(mnemonic)) = self.session.mnemonic_lookup(&ident) {
            if let TokenLineRule::LineEnd(()) =
                self.state.context.act_on_mnemonic(&mnemonic, span.clone())
            {
                return TokenLineRule::LineEnd(TokenContextFinalizationSemantics { parent: self });
            }
        }
        self.act_on_token(Token::Ident(ident), span);
        TokenLineRule::TokenSeq(self)
    }
}

pub(crate) trait ActOnMnemonic<M, S> {
    fn act_on_mnemonic(&mut self, mnemonic: M, span: S) -> TokenLineRule<(), ()>;
}

impl<R, S> ActOnMnemonic<&'static BuiltinMnemonic, S> for TokenContext<R, S>
where
    Self: ActOnToken<SemanticToken<R>, S>,
{
    fn act_on_mnemonic(
        &mut self,
        mnemonic: &'static BuiltinMnemonic,
        span: S,
    ) -> TokenLineRule<(), ()> {
        match (&*self, mnemonic) {
            (TokenContext::FalseIf, BuiltinMnemonic::Directive(Directive::Endc)) => {
                TokenLineRule::LineEnd(())
            }
            (TokenContext::MacroDef(_), BuiltinMnemonic::Directive(Directive::Endm)) => {
                self.act_on_token(Sigil::Eos.into(), span);
                TokenLineRule::LineEnd(())
            }
            _ => TokenLineRule::TokenSeq(()),
        }
    }
}

pub trait ActOnToken<T, S> {
    fn act_on_token(&mut self, token: T, span: S);
}

impl<R, S> ActOnToken<SemanticToken<R>, S> for TokenContext<R, S> {
    fn act_on_token(&mut self, token: SemanticToken<R>, span: S) {
        match self {
            TokenContext::FalseIf => drop((token, span)),
            TokenContext::MacroDef(state) => state.act_on_token(token, span),
        }
    }
}

impl<R, S> MacroDefState<R, S> {
    fn act_on_token(&mut self, token: SemanticToken<R>, span: S) {
        self.tokens.0.push(token);
        self.tokens.1.push(span)
    }
}

impl<'a, S: Analysis> LineFinalizer for TokenLineSemantics<'a, S> {
    type Next = TokenStreamSemantics<'a, S>;

    fn did_parse_line(mut self, span: S::Span) -> Self::Next {
        self.act_on_token(Sigil::Eol.into(), span);
        set_state!(self, self.state.into())
    }
}

pub(super) struct TokenContextFinalizationSemantics<'a, S: Analysis> {
    parent: TokenLineSemantics<'a, S>,
}

impl<'a, S: Analysis> ParsingContext for TokenContextFinalizationSemantics<'a, S> {
    type Ident = S::StringRef;
    type Literal = Literal<S::StringRef>;
    type Error = LexError;
    type Span = S::Span;
    type Stripped = <S as StripSpan<S::Span>>::Stripped;

    fn next_token(
        &mut self,
    ) -> Option<LexerOutput<Self::Ident, Self::Literal, Self::Error, Self::Span>> {
        self.parent.session.next_token()
    }

    fn merge_spans(&mut self, left: &Self::Span, right: &Self::Span) -> Self::Span {
        self.parent.session.merge_spans(left, right)
    }

    fn strip_span(&mut self, span: &Self::Span) -> Self::Stripped {
        self.parent.session.strip_span(span)
    }

    fn emit_diag(&mut self, diag: impl Into<CompactDiag<Self::Span, Self::Stripped>>) {
        self.parent.session.emit_diag(diag)
    }
}

impl<'a, S: Analysis> LineFinalizer for TokenContextFinalizationSemantics<'a, S> {
    type Next = TokenStreamSemantics<'a, S>;

    fn did_parse_line(self, _: S::Span) -> Self::Next {
        match self.parent.state.context {
            TokenContext::FalseIf => (),
            TokenContext::MacroDef(state) => {
                if let Some((name, params)) = state.label {
                    let tokens = state.tokens;
                    self.parent.session.define_macro(
                        name,
                        (params.0.into_boxed_slice(), params.1.into_boxed_slice()),
                        (tokens.0.into_boxed_slice(), tokens.1.into_boxed_slice()),
                    );
                }
            }
        }
        set_state!(self.parent, TokenStreamState::new())
    }
}

impl<'a, S: Analysis> InstrLineContext for InstrLineSemantics<'a, S>
where
    S::StringRef: 'static,
    S::Span: 'static,
{
    type LabelContext = LabelSemantics<'a, S>;
    type InstrContext = Self;

    fn will_parse_label(mut self, label: (S::StringRef, S::Span)) -> Self::LabelContext {
        self.flush_label();
        self.map_state(|line| LabelState::new(line, label))
    }
}

impl<'a, S: Analysis> InstrContext for InstrLineSemantics<'a, S>
where
    S::StringRef: 'static,
    S::Span: 'static,
{
    type BuiltinInstrContext = BuiltinInstrSemantics<'a, S>;
    type MacroInstrContext = MacroInstrSemantics<'a, S>;
    type ErrorContext = Self;
    type LineFinalizer = TokenStreamSemantics<'a, S>;

    fn will_parse_instr(
        mut self,
        ident: S::StringRef,
        span: S::Span,
    ) -> InstrRule<Self::BuiltinInstrContext, Self::MacroInstrContext, Self> {
        match self.session.mnemonic_lookup(&ident) {
            Some(MnemonicEntry::Builtin(mnemonic)) => {
                if !mnemonic.binds_to_label() {
                    self.flush_label();
                }
                InstrRule::BuiltinInstr(set_state!(
                    self,
                    BuiltinInstrState::new(self.state.label, (*mnemonic).clone().with_span(span))
                ))
            }
            Some(MnemonicEntry::Macro(id)) => {
                self.flush_label();
                InstrRule::MacroInstr(set_state!(
                    self,
                    MacroInstrState::new(self.state, (id, span))
                ))
            }
            None => {
                let name = self.session.strip_span(&span);
                self.session
                    .emit_diag(Message::NotAMnemonic { name }.at(span));
                InstrRule::Error(self)
            }
        }
    }
}

impl<'a, S: Analysis> InstrLineSemantics<'a, S> {
    pub fn flush_label(&mut self) {
        if let Some(((label, span), _params)) = self.state.label.take() {
            if self.session.name_visibility(&label) == Visibility::Global {
                self.session.start_scope();
            }
            let id = self.reloc_lookup(label, span.clone());
            self.session.define_symbol(
                id,
                span.clone(),
                Expr(vec![ExprOp::Atom(Atom::Location).with_span(span)]),
            );
        }
    }
}

impl<'a, S, T> Semantics<'a, S, T>
where
    S: Analysis,
{
    fn reloc_lookup(&mut self, name: S::StringRef, span: S::Span) -> SymbolId {
        match self.session.resolve_name(&name) {
            Some(ResolvedName::Keyword(_)) => unimplemented!(),
            Some(ResolvedName::Symbol(id)) => id,
            None => {
                let id = self.session.alloc_symbol(span);
                self.session.define_name(name, ResolvedName::Symbol(id));
                id
            }
        }
    }
}

pub(super) type LabelSemantics<'a, S> = Semantics<'a, S, LabelState<S>>;

pub(super) struct LabelState<S: Analysis> {
    parent: InstrLineState<S::StringRef, S::Span>,
    label: (S::StringRef, S::Span),
    params: Params<S::StringRef, S::Span>,
}

impl<S: Analysis> LabelState<S> {
    pub fn new(
        parent: InstrLineState<S::StringRef, S::Span>,
        label: (S::StringRef, S::Span),
    ) -> Self {
        Self {
            parent,
            label,
            params: (Vec::new(), Vec::new()),
        }
    }
}

impl<'a, S: Analysis> LabelContext for LabelSemantics<'a, S> {
    type Next = InstrLineSemantics<'a, S>;

    fn act_on_param(&mut self, ident: S::StringRef, span: S::Span) {
        let params = &mut self.state.params;
        params.0.push(ident);
        params.1.push(span)
    }

    fn did_parse_label(mut self) -> Self::Next {
        self.state.parent.label = Some((self.state.label, self.state.params));
        set_state!(self, self.state.parent)
    }
}

pub(super) type MacroInstrSemantics<'a, S> = Semantics<'a, S, MacroInstrState<S>>;

pub(super) struct MacroInstrState<S: Analysis> {
    parent: InstrLineState<S::StringRef, S::Span>,
    name: (MacroId, S::Span),
    args: (Vec<Box<[SemanticToken<S::StringRef>]>>, Vec<Box<[S::Span]>>),
}

impl<S: Analysis> MacroInstrState<S> {
    pub fn new(parent: InstrLineState<S::StringRef, S::Span>, name: (MacroId, S::Span)) -> Self {
        Self {
            parent,
            name,
            args: (Vec::new(), Vec::new()),
        }
    }

    fn push_arg(&mut self, arg: TokenSeq<S::StringRef, S::Span>) {
        let args = &mut self.args;
        args.0.push(arg.0.into_boxed_slice());
        args.1.push(arg.1.into_boxed_slice())
    }
}

impl<'a, S: Analysis> MacroInstrContext for MacroInstrSemantics<'a, S>
where
    S::StringRef: 'static,
    S::Span: 'static,
{
    type MacroArgContext = MacroArgSemantics<'a, S>;

    fn will_parse_macro_arg(self) -> Self::MacroArgContext {
        set_state!(self, MacroArgState::new(self.state))
    }
}

impl<'a, S: Analysis> InstrFinalizer for MacroInstrSemantics<'a, S>
where
    S::StringRef: 'static,
    S::Span: 'static,
{
    type Next = TokenStreamSemantics<'a, S>;

    fn did_parse_instr(self) -> Self::Next {
        self.session.expand_macro(
            self.state.name,
            (
                self.state.args.0.into_boxed_slice(),
                self.state.args.1.into_boxed_slice(),
            ),
        );
        Semantics {
            session: self.session,
            state: TokenStreamState::from(self.state.parent),
        }
    }
}

type MacroArgSemantics<'a, S> = Semantics<'a, S, MacroArgState<S>>;

pub(super) struct MacroArgState<S: Analysis> {
    tokens: TokenSeq<S::StringRef, S::Span>,
    parent: MacroInstrState<S>,
}

impl<S: Analysis> MacroArgState<S> {
    fn new(parent: MacroInstrState<S>) -> Self {
        Self {
            tokens: (Vec::new(), Vec::new()),
            parent,
        }
    }
}

impl<'a, S: Analysis> MacroArgContext for MacroArgSemantics<'a, S> {
    type Next = MacroInstrSemantics<'a, S>;

    fn act_on_token(&mut self, (token, span): (SemanticToken<S::StringRef>, S::Span)) {
        let tokens = &mut self.state.tokens;
        tokens.0.push(token);
        tokens.1.push(span)
    }

    fn did_parse_macro_arg(mut self) -> Self::Next {
        self.state.parent.push_arg(self.state.tokens);
        set_state!(self, self.state.parent)
    }
}

impl<S: Analysis> From<BuiltinInstrState<S>>
    for TokenStreamState<<S as StringSource>::StringRef, <S as SpanSource>::Span>
{
    fn from(_: BuiltinInstrState<S>) -> Self {
        InstrLineState::new().into()
    }
}

impl<'a, S: Analysis> BuiltinInstrContext for BuiltinInstrSemantics<'a, S>
where
    S::StringRef: 'static,
    S::Span: 'static,
{
    type ArgContext = ArgSemantics<'a, S>;

    fn will_parse_arg(self) -> Self::ArgContext {
        Semantics {
            session: self.session,
            state: ExprBuilder::new(self.state),
        }
    }
}

impl<'a, S: Analysis> InstrFinalizer for BuiltinInstrSemantics<'a, S>
where
    S::StringRef: 'static,
    S::Span: 'static,
{
    type Next = TokenStreamSemantics<'a, S>;

    fn did_parse_instr(self) -> Self::Next {
        let args = self.state.args;
        let session = set_state!(self, InstrLineState::new().into());
        match self.state.mnemonic.item {
            BuiltinMnemonic::CpuInstr(cpu_instr) => {
                analyze_mnemonic(
                    (&cpu_instr, self.state.mnemonic.span),
                    args,
                    session.session,
                );
                session.map_state(Into::into)
            }
            BuiltinMnemonic::Directive(directive) => directive::analyze_directive(
                (directive, self.state.mnemonic.span),
                self.state.label,
                args,
                session,
            ),
        }
    }
}

impl<'a, S: Analysis, T> Semantics<'a, S, T> {
    fn expect_const(
        &mut self,
        arg: ParsedArg<S::StringRef, S::Span>,
    ) -> Result<Expr<SymbolId, S::Span>, ()> {
        match self.session.resolve_names(arg)? {
            Arg::Bare(BareArg::Const(value)) => Ok(value),
            Arg::Bare(BareArg::Symbol(_, span)) => {
                let keyword = self.session.strip_span(&span);
                self.session
                    .emit_diag(Message::KeywordInExpr { keyword }.at(span));
                Err(())
            }
            Arg::Error => Err(()),
            _ => unimplemented!(),
        }
    }

    fn define_symbol_with_params(
        &mut self,
        (name, span): (S::StringRef, S::Span),
        expr: ParsedArg<S::StringRef, S::Span>,
    ) {
        if let Ok(expr) = self.expect_const(expr) {
            let id = self.reloc_lookup(name, span.clone());
            self.session.define_symbol(id, span, expr);
        }
    }
}

impl From<Directive> for BuiltinMnemonic {
    fn from(directive: Directive) -> Self {
        BuiltinMnemonic::Directive(directive)
    }
}

impl From<Mnemonic> for BuiltinMnemonic {
    fn from(mnemonic: Mnemonic) -> Self {
        BuiltinMnemonic::CpuInstr(mnemonic)
    }
}

fn analyze_mnemonic<S: Analysis>(
    name: (&Mnemonic, S::Span),
    args: BuiltinInstrArgs<S::StringRef, S::Span>,
    session: &mut S,
) {
    let mut operands = Vec::new();
    let mut error = false;
    for arg in args {
        if let Ok(arg) = session.resolve_names(arg) {
            let operand = cpu_instr::operand::analyze_operand(arg, name.0.context(), session);
            operands.push(operand)
        } else {
            error = true;
        }
    }
    if !error {
        cpu_instr::analyze_instruction(name, operands, session)
    }
}

trait Resolve<R, S> {
    fn resolve_names(&mut self, arg: ParsedArg<R, S>) -> Result<Arg<R, S>, ()>;
}

trait ClassifyExpr<I, S> {
    fn classify_expr(&mut self, expr: Expr<I, S>) -> Result<BareArg<S>, ()>;
}

impl<T, S> Resolve<T::StringRef, S> for T
where
    T: Interner + NameTable<<T as StringSource>::StringRef> + Diagnostics<S> + AllocSymbol<S>,
    S: Clone,
{
    fn resolve_names(
        &mut self,
        arg: ParsedArg<T::StringRef, S>,
    ) -> Result<Arg<T::StringRef, S>, ()> {
        match arg {
            ParsedArg::Bare(expr) => match self.classify_expr(expr)? {
                BareArg::Symbol(symbol, span) => Ok(Arg::Bare(BareArg::Symbol(symbol, span))),
                BareArg::Const(expr) => Ok(Arg::Bare(BareArg::Const(expr))),
            },
            ParsedArg::Parenthesized(expr, span) => match self.classify_expr(expr)? {
                BareArg::Symbol(symbol, inner_span) => {
                    Ok(Arg::Deref(BareArg::Symbol(symbol, inner_span), span))
                }
                BareArg::Const(expr) => Ok(Arg::Deref(BareArg::Const(expr), span)),
            },
            ParsedArg::String(string, span) => Ok(Arg::String(string, span)),
            ParsedArg::Error => Ok(Arg::Error),
        }
    }
}

impl<T, S> ClassifyExpr<T::StringRef, S> for T
where
    T: Interner + NameTable<<T as StringSource>::StringRef> + Diagnostics<S> + AllocSymbol<S>,
    S: Clone,
{
    fn classify_expr(&mut self, mut expr: Expr<T::StringRef, S>) -> Result<BareArg<S>, ()> {
        if expr.0.len() == 1 {
            let node = expr.0.pop().unwrap();
            match node.item {
                ExprOp::Atom(Atom::Name(name)) => match self.resolve_name(&name) {
                    Some(ResolvedName::Keyword(operand)) => Ok(BareArg::Symbol(operand, node.span)),
                    Some(ResolvedName::Symbol(id)) => {
                        Ok(BareArg::Const(Expr(vec![
                            ExprOp::Atom(Atom::Name(id)).with_span(node.span)
                        ])))
                    }
                    None => {
                        let id = self.alloc_symbol(node.span.clone());
                        self.define_name(name, ResolvedName::Symbol(id));
                        Ok(BareArg::Const(Expr(vec![
                            ExprOp::Atom(Atom::Name(id)).with_span(node.span)
                        ])))
                    }
                },
                ExprOp::Atom(Atom::Const(n)) => {
                    Ok(BareArg::Const(Expr(vec![
                        ExprOp::Atom(Atom::Const(n)).with_span(node.span)
                    ])))
                }
                ExprOp::Atom(Atom::Location) => {
                    Ok(BareArg::Const(Expr(vec![
                        ExprOp::Atom(Atom::Location).with_span(node.span)
                    ])))
                }
                ExprOp::Atom(Atom::Param(id)) => {
                    Ok(BareArg::Const(Expr(vec![
                        ExprOp::Atom(Atom::Param(id)).with_span(node.span)
                    ])))
                }
                _ => panic!("first node in expression must be an atom"),
            }
        } else {
            let mut nodes = Vec::new();
            let mut error = false;
            for node in expr.0 {
                match node.item {
                    ExprOp::Atom(Atom::Name(name)) => match self.resolve_name(&name) {
                        Some(ResolvedName::Keyword(_)) => {
                            let keyword = self.strip_span(&node.span);
                            self.emit_diag(Message::KeywordInExpr { keyword }.at(node.span));
                            error = true
                        }
                        Some(ResolvedName::Symbol(id)) => {
                            nodes.push(ExprOp::Atom(Atom::Name(id)).with_span(node.span))
                        }
                        None => {
                            let id = self.alloc_symbol(node.span.clone());
                            self.define_name(name, ResolvedName::Symbol(id));
                            nodes.push(ExprOp::Atom(Atom::Name(id)).with_span(node.span))
                        }
                    },
                    ExprOp::Atom(Atom::Const(n)) => {
                        nodes.push(ExprOp::Atom(Atom::Const(n)).with_span(node.span))
                    }
                    ExprOp::Atom(Atom::Location) => {
                        nodes.push(ExprOp::Atom(Atom::Location).with_span(node.span))
                    }
                    ExprOp::Atom(Atom::Param(id)) => {
                        nodes.push(ExprOp::Atom(Atom::Param(id)).with_span(node.span))
                    }
                    ExprOp::Binary(op) => nodes.push(ExprOp::Binary(op).with_span(node.span)),
                    ExprOp::FnCall(arity) => nodes.push(ExprOp::FnCall(arity).with_span(node.span)),
                }
            }
            if !error {
                Ok(BareArg::Const(Expr(nodes)))
            } else {
                Err(())
            }
        }
    }
}

impl<'a, S: Analysis> ArgFinalizer for ArgSemantics<'a, S> {
    type Next = BuiltinInstrSemantics<'a, S>;

    fn did_parse_arg(mut self) -> Self::Next {
        let arg = self.state.arg.unwrap_or(ParsedArg::Error);
        self.state.parent.args.push(arg);
        Semantics {
            session: self.session,
            state: self.state.parent,
        }
    }
}

impl<'a, S: Analysis> ArgContext for ArgSemantics<'a, S> {
    fn act_on_atom(&mut self, atom: ExprAtom<S::StringRef, Literal<S::StringRef>>, span: S::Span) {
        match atom {
            ExprAtom::Ident(ident) => self.act_on_ident(ident, span),
            ExprAtom::Literal(Literal::Number(n)) => {
                self.act_on_expr_node(ExprOp::Atom(Atom::Const(n)), span)
            }
            ExprAtom::Literal(Literal::String(string)) => {
                self.state.arg = Some(ParsedArg::String(string, span))
            }
            ExprAtom::LocationCounter => self.act_on_expr_node(ExprOp::Atom(Atom::Location), span),
            ExprAtom::Error => self.state.arg = Some(ParsedArg::Error),
        }
    }

    fn act_on_operator(&mut self, op: Operator, span: S::Span) {
        match op {
            Operator::Binary(op) => self.act_on_expr_node(ExprOp::Binary(op), span),
            Operator::FnCall(arity) => self.act_on_expr_node(ExprOp::FnCall(arity), span),
            Operator::Unary(UnaryOperator::Parentheses) => match &self.state.arg {
                Some(ParsedArg::Bare(arg)) => {
                    self.state.arg = Some(ParsedArg::Parenthesized((*arg).clone(), span))
                }
                _ => unimplemented!(),
            },
        }
    }
}

impl<'a, S: Analysis> ArgSemantics<'a, S> {
    fn act_on_expr_node(&mut self, node: ExprOp<S::StringRef>, span: S::Span) {
        self.state.arg = match self.state.arg.take() {
            None => Some(ParsedArg::Bare(Expr(vec![node.with_span(span)]))),
            Some(ParsedArg::Bare(mut expr)) | Some(ParsedArg::Parenthesized(mut expr, _)) => {
                expr.0.push(node.with_span(span));
                Some(ParsedArg::Bare(expr))
            }
            Some(ParsedArg::Error) => Some(ParsedArg::Error),
            Some(ParsedArg::String(_, _)) => todo!(),
        }
    }

    fn act_on_ident(&mut self, ident: S::StringRef, span: S::Span) {
        let no_params = vec![];
        let params = match &self.state.parent.label {
            Some((_, (params, _))) => &params,
            _ => &no_params,
        };
        let param = params.iter().position(|param| *param == ident).map(ParamId);
        match param {
            None => self.act_on_expr_node(ExprOp::Atom(Atom::Name(ident)), span),
            Some(id) => self.act_on_expr_node(ExprOp::Atom(Atom::Param(id)), span),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::assembler::session::mock::Expr;
    use crate::assembler::session::mock::*;
    use crate::assembler::session::MacroId;
    use crate::assembler::syntax::{SemanticToken, Sigil, Token};
    use crate::diagnostics::mock::{Merge, MockSpan};
    use crate::expr::{Atom, BinOp, ExprOp, LocationCounter};
    use crate::object::{Fragment, Symbol, SymbolId, UserDefId, Width};

    use std::borrow::Borrow;
    use std::fmt::Debug;

    pub(super) type Event<S> = crate::assembler::session::Event<SymbolId, MacroId, String, S, S>;

    #[test]
    fn ident_with_underscore_prefix_is_local() {
        let session = MockSession::<()>::default();
        assert_eq!(
            session.name_visibility(&"_loop".to_owned()),
            Visibility::Local
        )
    }

    #[test]
    fn ident_without_underscore_prefix_is_global() {
        let session = MockSession::<()>::default();
        assert_eq!(
            session.name_visibility(&"start".to_owned()),
            Visibility::Global
        )
    }

    #[test]
    fn emit_ld_b_deref_hl() {
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("LD".into(), ())
                .into_builtin_instr();
            let mut arg1 = command.will_parse_arg();
            arg1.act_on_atom(ExprAtom::Ident("B".into()), ());
            command = arg1.did_parse_arg();
            let mut arg2 = command.will_parse_arg();
            arg2.act_on_atom(ExprAtom::Ident("HL".into()), ());
            arg2.act_on_operator(Operator::Unary(UnaryOperator::Parentheses), ());
            arg2.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [Event::EmitFragment {
                fragment: Fragment::Byte(0x46)
            }]
        )
    }

    #[test]
    fn emit_rst_1_minus_1() {
        test_rst_1_op_1(BinOp::Minus)
    }

    #[test]
    fn emit_rst_1_plus_1() {
        test_rst_1_op_1(BinOp::Plus)
    }

    fn test_rst_1_op_1(op: BinOp) {
        use crate::expr::*;

        let actions = collect_semantic_actions(|actions| {
            let command = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("RST".into(), ())
                .into_builtin_instr();
            let mut expr = command.will_parse_arg();
            expr.act_on_atom(ExprAtom::Literal(Literal::Number(1)), ());
            expr.act_on_atom(ExprAtom::Literal(Literal::Number(1)), ());
            expr.act_on_operator(Operator::Binary(op), ());
            expr.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [Event::EmitFragment {
                fragment: Fragment::Embedded(
                    0b11_000_111,
                    Expr::from_items(&[1.into(), 1.into(), op.into()])
                )
            }]
        )
    }

    #[test]
    fn emit_rst_f_of_1() {
        let name = "f";
        let actions = collect_semantic_actions(|actions| {
            let command = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("RST".into(), ())
                .into_builtin_instr();
            let mut expr = command.will_parse_arg();
            expr.act_on_atom(ExprAtom::Ident(name.to_owned()), ());
            expr.act_on_atom(ExprAtom::Literal(Literal::Number(1)), ());
            expr.act_on_operator(Operator::FnCall(1), ());
            expr.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [
                Event::DefineNameWithVisibility {
                    ident: name.to_owned(),
                    visibility: Visibility::Global,
                    entry: ResolvedName::Symbol(Symbol::UserDef(UserDefId(0))),
                },
                Event::EmitFragment {
                    fragment: Fragment::Embedded(
                        0b11_000_111,
                        Expr::from_items(&[
                            Atom::Name(Symbol::UserDef(UserDefId(0))).into(),
                            1.into(),
                            ExprOp::FnCall(1).into()
                        ])
                    )
                }
            ]
        )
    }

    #[test]
    fn emit_label_word() {
        let label = "my_label";
        let actions = collect_semantic_actions(|actions| {
            let mut arg = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("DW".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            arg.act_on_atom(ExprAtom::Ident(label.into()), ());
            arg.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [
                Event::DefineNameWithVisibility {
                    ident: label.into(),
                    visibility: Visibility::Global,
                    entry: ResolvedName::Symbol(Symbol::UserDef(UserDefId(0)))
                },
                Event::EmitFragment {
                    fragment: Fragment::Immediate(
                        Atom::Name(Symbol::UserDef(UserDefId(0))).into(),
                        Width::Word
                    )
                }
            ]
        );
    }

    #[test]
    fn analyze_label() {
        let label = "label";
        let actions = collect_semantic_actions(|actions| {
            actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_label((label.into(), ()))
                .did_parse_label()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [
                Event::StartScope,
                Event::DefineNameWithVisibility {
                    ident: label.into(),
                    visibility: Visibility::Global,
                    entry: ResolvedName::Symbol(Symbol::UserDef(UserDefId(0)))
                },
                Event::DefineSymbol {
                    name: Symbol::UserDef(UserDefId(0)),
                    span: (),
                    expr: LocationCounter.into()
                }
            ]
        )
    }

    #[test]
    fn analyze_org_dot() {
        let actions = collect_semantic_actions(|actions| {
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("ORG".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            actions.act_on_atom(ExprAtom::LocationCounter, ());
            actions
                .did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [Event::SetOrigin {
                addr: LocationCounter.into()
            }]
        );
    }

    #[test]
    fn define_nullary_macro() {
        test_macro_definition(
            "my_macro",
            [],
            [Token::Ident("XOR".into()), Token::Ident("A".into())],
        )
    }

    #[test]
    fn define_unary_macro() {
        let param = "reg";
        test_macro_definition(
            "my_xor",
            [param],
            [Token::Ident("XOR".into()), Token::Ident(param.into())],
        )
    }

    #[test]
    fn define_nameless_macro() {
        let actions = collect_semantic_actions(|actions| {
            actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("MACRO".into(), ())
                .into_builtin_instr()
                .did_parse_instr()
                .did_parse_line(())
                .will_parse_line()
                .into_token_line()
                .act_on_mnemonic("ENDM".into(), ())
                .into_line_end()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [Event::EmitDiag {
                diag: Message::MacroRequiresName.at(()).into()
            }]
        )
    }

    fn test_macro_definition(
        name: &str,
        params: impl Borrow<[&'static str]>,
        body: impl Borrow<[SemanticToken<String>]>,
    ) {
        let actions = collect_semantic_actions(|actions| {
            let mut params_actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_label((name.into(), ()));
            for param in params.borrow().iter().cloned().map(Into::into) {
                params_actions.act_on_param(param, ())
            }
            let mut token_seq_actions = params_actions
                .did_parse_label()
                .will_parse_instr("MACRO".into(), ())
                .into_builtin_instr()
                .did_parse_instr()
                .did_parse_line(())
                .will_parse_line()
                .into_token_line();
            for token in body.borrow().iter().cloned() {
                token_seq_actions.act_on_token(token, ())
            }
            token_seq_actions
                .act_on_mnemonic("ENDM".into(), ())
                .into_line_end()
                .did_parse_line(())
                .act_on_eos(())
        });
        let params = params
            .borrow()
            .iter()
            .cloned()
            .map(Into::into)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let param_spans = vec![(); params.len()].into_boxed_slice();
        let mut body = body.borrow().to_vec();
        body.push(Sigil::Eos.into());
        let body_spans = vec![(); body.len()].into_boxed_slice();
        assert_eq!(
            actions,
            [Event::DefineMacro {
                name: (name.into(), ()),
                params: (params, param_spans),
                body: (body.into_boxed_slice(), body_spans),
            }]
        )
    }

    #[test]
    fn diagnose_parsing_error() {
        let diagnostic = Message::UnexpectedToken { token: () }.at(());
        let actions = collect_semantic_actions(|mut actions| {
            actions.emit_diag(diagnostic.clone());
            actions.did_parse_line(()).act_on_eos(())
        });
        assert_eq!(
            actions,
            [Event::EmitDiag {
                diag: diagnostic.into()
            }]
        )
    }

    #[test]
    fn recover_from_malformed_expr() {
        let diagnostic = Message::UnexpectedToken { token: () }.at(());
        let actions = collect_semantic_actions(|file| {
            let mut expr = file
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("ADD".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            expr.act_on_atom(ExprAtom::Error, ());
            expr.emit_diag(diagnostic.clone());
            expr.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [Event::EmitDiag {
                diag: diagnostic.into()
            }]
        )
    }

    #[test]
    fn diagnose_eos_in_macro_body() {
        let log = collect_semantic_actions::<_, MockSpan<_>>(|actions| {
            actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_label(("my_macro".into(), "label".into()))
                .did_parse_label()
                .will_parse_instr("MACRO".into(), "key".into())
                .into_builtin_instr()
                .did_parse_instr()
                .did_parse_line("eol".into())
                .will_parse_line()
                .into_token_line()
                .did_parse_line("eos".into())
                .act_on_eos("eos".into())
        });
        assert_eq!(
            log,
            [Event::EmitDiag {
                diag: Message::UnexpectedEof.at("eos".into()).into()
            }]
        )
    }

    pub(super) fn collect_semantic_actions<F, S>(f: F) -> Vec<Event<S>>
    where
        F: for<'a> FnOnce(TestTokenStreamSemantics<'a, S>) -> TestTokenStreamSemantics<'a, S>,
        S: Clone + Default + Debug + Merge,
    {
        log_with_predefined_names(std::iter::empty(), f)
    }

    pub(super) fn log_with_predefined_names<I, F, S>(entries: I, f: F) -> Vec<Event<S>>
    where
        I: IntoIterator<Item = (String, ResolvedName)>,
        F: for<'a> FnOnce(TestTokenStreamSemantics<'a, S>) -> TestTokenStreamSemantics<'a, S>,
        S: Clone + Default + Debug + Merge,
    {
        let mut session = MockSession::default();
        for (ident, resolution) in entries {
            session.define_name(ident, resolution)
        }
        f(Semantics {
            session: &mut session,
            state: TokenStreamState::new(),
        });
        session.log().to_vec()
    }

    pub(super) type TestTokenStreamSemantics<'a, S> = TokenStreamSemantics<'a, MockSession<S>>;

    #[test]
    fn diagnose_unknown_mnemonic() {
        let name = "unknown";
        let log = collect_semantic_actions::<_, MockSpan<_>>(|session| {
            session
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr(name.into(), name.into())
                .error()
                .unwrap()
                .did_parse_instr()
                .did_parse_line("eol".into())
                .act_on_eos("eos".into())
        });
        assert_eq!(
            log,
            [Event::EmitDiag {
                diag: Message::NotAMnemonic { name: name.into() }
                    .at(name.into())
                    .into()
            }]
        )
    }

    #[test]
    fn diagnose_operand_as_mnemonic() {
        let name = "HL";
        let log = collect_semantic_actions::<_, MockSpan<_>>(|session| {
            session
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr(name.into(), name.into())
                .error()
                .unwrap()
                .did_parse_instr()
                .did_parse_line("eol".into())
                .act_on_eos("eos".into())
        });
        assert_eq!(
            log,
            [Event::EmitDiag {
                diag: Message::NotAMnemonic { name: name.into() }
                    .at(name.into())
                    .into()
            }]
        )
    }

    #[test]
    fn call_nullary_macro() {
        let name = "my_macro";
        let macro_id = MacroId(0);
        let log = collect_semantic_actions(|actions| {
            actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_label((name.into(), ()))
                .did_parse_label()
                .will_parse_instr("MACRO".into(), ())
                .into_builtin_instr()
                .did_parse_instr()
                .did_parse_line(())
                .will_parse_line()
                .into_token_line()
                .act_on_mnemonic("ENDM".into(), ())
                .into_line_end()
                .did_parse_line(())
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr(name.into(), ())
                .into_macro_instr()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            log,
            [
                Event::DefineMacro {
                    name: (name.into(), ()),
                    params: (Box::new([]), Box::new([])),
                    body: (Box::new([Token::Sigil(Sigil::Eos)]), Box::new([()])),
                },
                Event::ExpandMacro {
                    name: (macro_id, ()),
                    args: (Box::new([]), Box::new([]))
                }
            ]
        )
    }

    #[test]
    fn call_unary_macro() {
        let name = "my_macro";
        let arg_token = Token::Ident("A".into());
        let macro_id = MacroId(0);
        let log = collect_semantic_actions(|actions| {
            let mut params = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_label((name.into(), ()));
            params.act_on_param("param".into(), ());
            let mut call = params
                .did_parse_label()
                .will_parse_instr("MACRO".into(), ())
                .into_builtin_instr()
                .did_parse_instr()
                .did_parse_line(())
                .will_parse_line()
                .into_token_line()
                .act_on_mnemonic("ENDM".into(), ())
                .into_line_end()
                .did_parse_line(())
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr(name.into(), ())
                .into_macro_instr();
            call = {
                let mut arg = call.will_parse_macro_arg();
                arg.act_on_token((arg_token.clone(), ()));
                arg.did_parse_macro_arg()
            };
            call.did_parse_instr().did_parse_line(()).act_on_eos(())
        });
        assert_eq!(
            log,
            [
                Event::DefineMacro {
                    name: (name.into(), ()),
                    params: (Box::new(["param".into()]), Box::new([()])),
                    body: (Box::new([Token::Sigil(Sigil::Eos)]), Box::new([()])),
                },
                Event::ExpandMacro {
                    name: (macro_id, ()),
                    args: (
                        vec![vec![arg_token].into_boxed_slice()].into_boxed_slice(),
                        Box::new([Box::new([()])])
                    ),
                }
            ]
        )
    }

    #[ignore]
    #[test]
    fn diagnose_literal_as_fn_name() {
        assert_eq!(
            collect_semantic_actions::<_, MockSpan<_>>(|actions| {
                let mut actions = actions
                    .will_parse_line()
                    .into_instr_line()
                    .will_parse_instr("DB".into(), "db".into())
                    .into_builtin_instr()
                    .will_parse_arg();
                actions.act_on_atom(ExprAtom::Literal(Literal::Number(7)), "literal".into());
                actions.act_on_operator(Operator::FnCall(0), "call".into());
                actions
                    .did_parse_arg()
                    .did_parse_instr()
                    .did_parse_line("eol".into())
                    .act_on_eos("eos".into())
            }),
            [Event::EmitDiag {
                diag: Message::OnlyIdentsCanBeCalled.at("literal".into()).into()
            }]
        );
    }

    #[test]
    fn diagnose_keyword_in_expr() {
        assert_eq!(
            collect_semantic_actions::<_, MockSpan<_>>(|actions| {
                let mut actions = actions
                    .will_parse_line()
                    .into_instr_line()
                    .will_parse_instr("DB".into(), "db".into())
                    .into_builtin_instr()
                    .will_parse_arg();
                actions.act_on_atom(ExprAtom::Ident("A".into()), "keyword".into());
                actions.act_on_atom(ExprAtom::Literal(Literal::Number(1)), "one".into());
                actions.act_on_operator(Operator::Binary(BinOp::Plus), "plus".into());
                actions
                    .did_parse_arg()
                    .did_parse_instr()
                    .did_parse_line("eol".into())
                    .act_on_eos("eos".into())
            }),
            [Event::EmitDiag {
                diag: Message::KeywordInExpr {
                    keyword: "keyword".into()
                }
                .at("keyword".into())
                .into()
            }]
        )
    }

    #[test]
    fn handle_unknown_name() {
        let actual = collect_semantic_actions::<_, MockSpan<_>>(|actions| {
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("DB".into(), "db".into())
                .into_builtin_instr()
                .will_parse_arg();
            actions.act_on_atom(ExprAtom::Ident("f".into()), "f".into());
            actions
                .did_parse_arg()
                .did_parse_instr()
                .did_parse_line("eol".into())
                .act_on_eos("eos".into())
        });
        let expected = [
            Event::DefineNameWithVisibility {
                ident: "f".into(),
                visibility: Visibility::Global,
                entry: ResolvedName::Symbol(Symbol::UserDef(UserDefId(0))),
            },
            Event::EmitFragment {
                fragment: Fragment::Immediate(
                    Expr::from_atom(Atom::Name(Symbol::UserDef(UserDefId(0))), "f".into()),
                    Width::Byte,
                ),
            },
        ];
        assert_eq!(actual, expected)
    }

    #[test]
    fn act_on_known_symbol_name() {
        let actual = collect_semantic_actions::<_, MockSpan<_>>(|actions| {
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("DB".into(), "db".into())
                .into_builtin_instr()
                .will_parse_arg();
            actions.act_on_atom(ExprAtom::Ident("f".into()), "f1".into());
            actions.act_on_atom(ExprAtom::Ident("f".into()), "f2".into());
            actions.act_on_operator(Operator::Binary(BinOp::Plus), "plus".into());
            actions
                .did_parse_arg()
                .did_parse_instr()
                .did_parse_line("eol".into())
                .act_on_eos("eos".into())
        });
        let expected = [
            Event::DefineNameWithVisibility {
                ident: "f".into(),
                visibility: Visibility::Global,
                entry: ResolvedName::Symbol(Symbol::UserDef(UserDefId(0))),
            },
            Event::EmitFragment {
                fragment: Fragment::Immediate(
                    Expr(vec![
                        ExprOp::Atom(Atom::Name(Symbol::UserDef(UserDefId(0))))
                            .with_span("f1".into()),
                        ExprOp::Atom(Atom::Name(Symbol::UserDef(UserDefId(0))))
                            .with_span("f2".into()),
                        ExprOp::Binary(BinOp::Plus).with_span("plus".into()),
                    ]),
                    Width::Byte,
                ),
            },
        ];
        assert_eq!(actual, expected)
    }

    #[test]
    fn handle_deref_const() {
        let actual = collect_semantic_actions::<_, MockSpan<_>>(|actions| {
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("LD".into(), "ld".into())
                .into_builtin_instr()
                .will_parse_arg();
            actions.act_on_atom(ExprAtom::Ident("A".into()), "a".into());
            let mut actions = actions.did_parse_arg().will_parse_arg();
            actions.act_on_atom(ExprAtom::Ident("const".into()), "const".into());
            actions.act_on_operator(Operator::Unary(UnaryOperator::Parentheses), "deref".into());
            actions
                .did_parse_arg()
                .did_parse_instr()
                .did_parse_line("eol".into())
                .act_on_eos("eos".into())
        });
        let expected = [
            Event::DefineNameWithVisibility {
                ident: "const".into(),
                visibility: Visibility::Global,
                entry: ResolvedName::Symbol(Symbol::UserDef(UserDefId(0))),
            },
            Event::EmitFragment {
                fragment: Fragment::LdInlineAddr(
                    0xf0,
                    Expr::from_atom(Atom::Name(Symbol::UserDef(UserDefId(0))), "const".into()),
                ),
            },
        ];
        assert_eq!(actual, expected)
    }

    #[test]
    fn handle_param() {
        let actual = collect_semantic_actions::<_, MockSpan<_>>(|context| {
            let mut context = context
                .will_parse_line()
                .into_instr_line()
                .will_parse_label(("label".into(), "label".into()));
            context.act_on_param("param".into(), "param1".into());
            let mut context = context
                .did_parse_label()
                .will_parse_instr("EQU".into(), "equ".into())
                .into_builtin_instr()
                .will_parse_arg();
            context.act_on_atom(ExprAtom::Ident("param".into()), "param2".into());
            context
                .did_parse_arg()
                .did_parse_instr()
                .did_parse_line("eol".into())
        });
        let expected = [
            Event::DefineNameWithVisibility {
                ident: "label".into(),
                visibility: Visibility::Global,
                entry: ResolvedName::Symbol(Symbol::UserDef(UserDefId(0))),
            },
            Event::DefineSymbol {
                name: Symbol::UserDef(UserDefId(0)),
                span: "label".into(),
                expr: Expr::from_atom(Atom::Param(ParamId(0)), "param2".into()),
            },
        ];
        assert_eq!(actual, expected)
    }
}
