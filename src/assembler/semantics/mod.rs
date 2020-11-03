use super::keywords::{BuiltinMnemonic, Directive, Mnemonic, OperandKeyword};
use super::session::*;
use super::string_ref::StringRef;
use super::syntax::actions::*;
use super::syntax::{LexError, Literal, SemanticToken, Sigil, Token};

use crate::diagnostics::{CompactDiag, Diagnostics, Message};
use crate::expr::{Atom, Expr, ExprOp, ParamId};
use crate::object::Name;
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

pub(super) struct Semantics<'a, S, T> {
    session: &'a mut S,
    state: T,
}

pub(super) trait SemanticActions<'a> {
    type SemanticActions: TokenStreamContext;
    fn semantic_actions(&'a mut self) -> Self::SemanticActions;
}

impl<'a, T: Analysis + 'a> SemanticActions<'a> for T
where
    T::Span: 'static,
{
    type SemanticActions = Semantics<'a, T, TokenStreamState<T::Span>>;

    fn semantic_actions(&'a mut self) -> Self::SemanticActions {
        Semantics {
            session: self,
            state: TokenStreamState::new(),
        }
    }
}

impl<'a, 'b, S: Analysis, T> Semantics<'a, S, T> {
    fn map_state<F: FnOnce(T) -> U, U>(self, f: F) -> Semantics<'a, S, U> {
        Semantics {
            session: self.session,
            state: f(self.state),
        }
    }
}

type TokenStreamSemantics<'a, S> = Semantics<'a, S, TokenStreamState<<S as SpanSource>::Span>>;

#[derive(Debug, PartialEq)]
pub(super) struct TokenStreamState<S> {
    mode: LineRule<InstrLineState<S>, TokenLineState<S>>,
}

impl<S> TokenStreamState<S> {
    pub fn new() -> Self {
        Self {
            mode: LineRule::InstrLine(InstrLineState::new()),
        }
    }
}

type InstrLineSemantics<'a, S> = Semantics<'a, S, InstrLineState<<S as SpanSource>::Span>>;

#[derive(Debug, PartialEq)]
pub struct InstrLineState<S> {
    label: Option<Label<S>>,
}

impl<S> InstrLineState<S> {
    fn new() -> Self {
        Self { label: None }
    }
}

type Label<S> = ((StringRef, S), Params<S>);
type Params<S> = (Vec<StringRef>, Vec<S>);

type TokenLineSemantics<'a, S> = Semantics<'a, S, TokenLineState<<S as SpanSource>::Span>>;

#[derive(Debug, PartialEq)]
pub struct TokenLineState<S> {
    context: TokenContext<S>,
}

#[derive(Debug, PartialEq)]
pub enum TokenContext<S> {
    FalseIf,
    MacroDef(MacroDefState<S>),
}

#[derive(Debug, PartialEq)]
pub struct MacroDefState<S> {
    label: Option<Label<S>>,
    tokens: TokenSeq<S>,
}

pub type TokenSeq<S> = (Vec<SemanticToken>, Vec<S>);

impl<S> MacroDefState<S> {
    fn new(label: Option<Label<S>>) -> Self {
        Self {
            label,
            tokens: (Vec::new(), Vec::new()),
        }
    }
}

type BuiltinInstrSemantics<'a, S> = Semantics<'a, S, BuiltinInstrState<S>>;

pub(super) struct BuiltinInstrState<S: Analysis> {
    label: Option<Label<S::Span>>,
    mnemonic: Spanned<BuiltinMnemonic, S::Span>,
    args: BuiltinInstrArgs<S::Span>,
}

impl<S: Analysis> BuiltinInstrState<S> {
    fn new(label: Option<Label<S::Span>>, mnemonic: Spanned<BuiltinMnemonic, S::Span>) -> Self {
        Self {
            label,
            mnemonic,
            args: Vec::new(),
        }
    }
}

type BuiltinInstrArgs<S> = Vec<ParsedArg<S>>;

pub(super) type ArgSemantics<'a, S> =
    Semantics<'a, S, ExprBuilder<<S as SpanSource>::Span, BuiltinInstrState<S>>>;

pub(crate) struct ExprBuilder<S, P> {
    arg: Option<ParsedArg<S>>,
    parent: P,
}

impl<S, P> ExprBuilder<S, P> {
    pub fn new(parent: P) -> Self {
        Self { arg: None, parent }
    }
}

enum ParsedArg<S> {
    Bare(Expr<StringRef, S>),
    Parenthesized(Expr<StringRef, S>, S),
    String(StringRef, S),
    Error,
}

enum Arg<S> {
    Bare(BareArg<S>),
    Deref(BareArg<S>, S),
    String(StringRef, S),
    Error,
}

#[derive(Clone)]
enum BareArg<S> {
    Const(Expr<Name, S>),
    OperandKeyword(OperandKeyword, S),
}

trait NameVisibility {
    fn name_visibility(&self, name: &str) -> Visibility;
}

impl<S> NameVisibility for S {
    fn name_visibility(&self, name: &str) -> Visibility {
        if name.starts_with('_') {
            Visibility::Local
        } else {
            Visibility::Global
        }
    }
}

impl<S> From<InstrLineState<S>> for TokenStreamState<S> {
    fn from(actions: InstrLineState<S>) -> Self {
        Self {
            mode: LineRule::InstrLine(actions),
        }
    }
}

impl<S> From<TokenLineState<S>> for TokenStreamState<S> {
    fn from(actions: TokenLineState<S>) -> Self {
        Self {
            mode: LineRule::TokenLine(actions),
        }
    }
}

impl<'a, S, T> ParsingContext for Semantics<'a, S, T>
where
    S: NextToken + Diagnostics<<S as SpanSource>::Span>,
{
    type Ident = StringRef;
    type Literal = Literal;
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

    fn act_on_token(&mut self, token: SemanticToken, span: S::Span) {
        match &mut self.state.context {
            TokenContext::FalseIf => (),
            TokenContext::MacroDef(state) => state.act_on_token(token, span),
        }
    }

    fn act_on_mnemonic(
        mut self,
        ident: StringRef,
        span: S::Span,
    ) -> TokenLineRule<Self, Self::ContextFinalizer> {
        if let Some(MnemonicEntry::Builtin(mnemonic)) = self.session.mnemonic_lookup(ident.clone())
        {
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

impl<S> ActOnMnemonic<&'static BuiltinMnemonic, S> for TokenContext<S>
where
    Self: ActOnToken<S>,
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

pub trait ActOnToken<S> {
    fn act_on_token(&mut self, token: SemanticToken, span: S);
}

impl<S> ActOnToken<S> for TokenContext<S> {
    fn act_on_token(&mut self, token: SemanticToken, span: S) {
        match self {
            TokenContext::FalseIf => drop((token, span)),
            TokenContext::MacroDef(state) => state.act_on_token(token, span),
        }
    }
}

impl<S> MacroDefState<S> {
    fn act_on_token(&mut self, token: SemanticToken, span: S) {
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
    type Ident = StringRef;
    type Literal = Literal;
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
    S::Span: 'static,
{
    type LabelContext = LabelSemantics<'a, S>;
    type InstrContext = Self;

    fn will_parse_label(mut self, label: (StringRef, S::Span)) -> Self::LabelContext {
        self.flush_label();
        self.map_state(|line| LabelState::new(line, label))
    }
}

impl<'a, S: Analysis> InstrContext for InstrLineSemantics<'a, S>
where
    S::Span: 'static,
{
    type BuiltinInstrContext = BuiltinInstrSemantics<'a, S>;
    type MacroInstrContext = MacroInstrSemantics<'a, S>;
    type ErrorContext = Self;
    type LineFinalizer = TokenStreamSemantics<'a, S>;

    fn will_parse_instr(
        mut self,
        ident: StringRef,
        span: S::Span,
    ) -> InstrRule<Self::BuiltinInstrContext, Self::MacroInstrContext, Self> {
        match self.session.mnemonic_lookup(ident) {
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
            self.session.define_symbol(
                (label, span.clone()),
                SymbolDef::Closure(Expr(vec![ExprOp::Atom(Atom::Location).with_span(span)])),
            )
        }
    }
}

pub(super) type LabelSemantics<'a, S> = Semantics<'a, S, LabelState<S>>;

pub(super) struct LabelState<S: Analysis> {
    parent: InstrLineState<S::Span>,
    label: (StringRef, S::Span),
    params: Params<S::Span>,
}

impl<S: Analysis> LabelState<S> {
    pub fn new(parent: InstrLineState<S::Span>, label: (StringRef, S::Span)) -> Self {
        Self {
            parent,
            label,
            params: (Vec::new(), Vec::new()),
        }
    }
}

impl<'a, S: Analysis> LabelContext for LabelSemantics<'a, S> {
    type Next = InstrLineSemantics<'a, S>;

    fn act_on_param(&mut self, ident: StringRef, span: S::Span) {
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
    parent: InstrLineState<S::Span>,
    name: (MacroId, S::Span),
    args: (Vec<Box<[SemanticToken]>>, Vec<Box<[S::Span]>>),
}

impl<S: Analysis> MacroInstrState<S> {
    pub fn new(parent: InstrLineState<S::Span>, name: (MacroId, S::Span)) -> Self {
        Self {
            parent,
            name,
            args: (Vec::new(), Vec::new()),
        }
    }

    fn push_arg(&mut self, arg: TokenSeq<S::Span>) {
        let args = &mut self.args;
        args.0.push(arg.0.into_boxed_slice());
        args.1.push(arg.1.into_boxed_slice())
    }
}

impl<'a, S: Analysis> MacroInstrContext for MacroInstrSemantics<'a, S>
where
    S::Span: 'static,
{
    type MacroArgContext = MacroArgSemantics<'a, S>;

    fn will_parse_macro_arg(self) -> Self::MacroArgContext {
        set_state!(self, MacroArgState::new(self.state))
    }
}

impl<'a, S: Analysis> InstrFinalizer for MacroInstrSemantics<'a, S>
where
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
    tokens: TokenSeq<S::Span>,
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

    fn act_on_token(&mut self, (token, span): (SemanticToken, S::Span)) {
        let tokens = &mut self.state.tokens;
        tokens.0.push(token);
        tokens.1.push(span)
    }

    fn did_parse_macro_arg(mut self) -> Self::Next {
        self.state.parent.push_arg(self.state.tokens);
        set_state!(self, self.state.parent)
    }
}

impl<S: Analysis> From<BuiltinInstrState<S>> for TokenStreamState<<S as SpanSource>::Span> {
    fn from(_: BuiltinInstrState<S>) -> Self {
        InstrLineState::new().into()
    }
}

impl<'a, S: Analysis> BuiltinInstrContext for BuiltinInstrSemantics<'a, S>
where
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
    fn expect_const(&mut self, arg: ParsedArg<S::Span>) -> Result<Expr<Name, S::Span>, ()> {
        match self.session.resolve_names(arg)? {
            Arg::Bare(BareArg::Const(value)) => Ok(value),
            Arg::Bare(BareArg::OperandKeyword(_, span)) => {
                let keyword = self.session.strip_span(&span);
                self.session
                    .emit_diag(Message::KeywordInExpr { keyword }.at(span));
                Err(())
            }
            Arg::Error => Err(()),
            _ => unimplemented!(),
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
    args: BuiltinInstrArgs<S::Span>,
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

trait Resolve<S> {
    fn resolve_names(&mut self, arg: ParsedArg<S>) -> Result<Arg<S>, ()>;
}

trait ClassifyExpr<I, S> {
    fn classify_expr(&mut self, expr: Expr<I, S>) -> Result<BareArg<S>, ()>;
}

impl<T, S> Resolve<S> for T
where
    T: IdentTable + Diagnostics<S>,
    S: Clone,
{
    fn resolve_names(&mut self, arg: ParsedArg<S>) -> Result<Arg<S>, ()> {
        Ok(match arg {
            ParsedArg::Bare(expr) => Arg::Bare(self.classify_expr(expr)?),
            ParsedArg::Parenthesized(expr, span) => Arg::Deref(self.classify_expr(expr)?, span),
            ParsedArg::String(string, span) => Arg::String(string, span),
            ParsedArg::Error => Arg::Error,
        })
    }
}

impl<T, S> ClassifyExpr<StringRef, S> for T
where
    T: IdentTable + Diagnostics<S>,
    S: Clone,
{
    fn classify_expr(&mut self, mut expr: Expr<StringRef, S>) -> Result<BareArg<S>, ()> {
        if expr.0.len() == 1 {
            let node = expr.0.pop().unwrap();
            match node.item {
                ExprOp::Atom(Atom::Name(name)) => match self.query_term(&name) {
                    NameEntry::OperandKeyword(operand) => {
                        Ok(BareArg::OperandKeyword(operand, node.span))
                    }
                    NameEntry::Symbol(id) => {
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
                    ExprOp::Atom(Atom::Name(name)) => match self.query_term(&name) {
                        NameEntry::OperandKeyword(_) => {
                            let keyword = self.strip_span(&node.span);
                            self.emit_diag(Message::KeywordInExpr { keyword }.at(node.span));
                            error = true
                        }
                        NameEntry::Symbol(id) => {
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
    fn act_on_atom(&mut self, atom: ExprAtom<StringRef, Literal>, span: S::Span) {
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
    fn act_on_expr_node(&mut self, node: ExprOp<StringRef>, span: S::Span) {
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

    fn act_on_ident(&mut self, ident: StringRef, span: S::Span) {
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
    use crate::assembler::session::MacroId;
    use crate::assembler::syntax::{SemanticToken, Sigil, Token};
    use crate::diagnostics::mock::MockSpan;
    use crate::expr::{Atom, BinOp, ExprOp, LocationCounter};
    use crate::object::{Fragment, Name, SymbolId, Width};

    use std::borrow::Borrow;

    pub(super) type Event<S> = crate::assembler::session::Event<Name, MacroId, S, S>;

    #[test]
    fn ident_with_underscore_prefix_is_local() {
        let mut fixture = TestFixture::<()>::new();
        let session = fixture.session();
        assert_eq!(session.name_visibility("_loop"), Visibility::Local)
    }

    #[test]
    fn ident_without_underscore_prefix_is_global() {
        let mut fixture = TestFixture::<()>::new();
        let session = fixture.session();
        assert_eq!(session.name_visibility("start"), Visibility::Global)
    }

    #[test]
    fn emit_ld_b_deref_hl() {
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
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
                .act_on_eos(());
        }
        assert_eq!(
            session.log(),
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
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
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
                .act_on_eos(());
        }
        assert_eq!(
            session.log(),
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
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            let command = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("RST".into(), ())
                .into_builtin_instr();
            let mut expr = command.will_parse_arg();
            expr.act_on_atom(ExprAtom::Ident(name.into()), ());
            expr.act_on_atom(ExprAtom::Literal(Literal::Number(1)), ());
            expr.act_on_operator(Operator::FnCall(1), ());
            expr.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(());
        }
        assert_eq!(
            session.log(),
            [Event::EmitFragment {
                fragment: Fragment::Embedded(
                    0b11_000_111,
                    Expr::from_items(&[
                        Atom::Name(Name::Symbol(SymbolId(0))).into(),
                        1.into(),
                        ExprOp::FnCall(1).into()
                    ])
                )
            }]
        )
    }

    #[test]
    fn emit_label_word() {
        let label = "my_label";
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
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
                .act_on_eos(());
        }
        assert_eq!(
            session.log(),
            [Event::EmitFragment {
                fragment: Fragment::Immediate(
                    Atom::Name(Name::Symbol(SymbolId(0))).into(),
                    Width::Word
                )
            }]
        );
    }

    #[test]
    fn analyze_label() {
        let label = "label";
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_label((label.into(), ()))
                .did_parse_label()
                .did_parse_line(())
                .act_on_eos(());
        }
        assert_eq!(
            session.log(),
            [Event::DefineSymbol {
                symbol: (label.into(), ()),
                def: SymbolDef::Closure(LocationCounter.into()),
            }]
        )
    }

    #[test]
    fn analyze_org_dot() {
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
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
                .act_on_eos(());
        }
        assert_eq!(
            session.log(),
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
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
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
                .act_on_eos(());
        }
        assert_eq!(
            session.log(),
            [Event::EmitDiag {
                diag: Message::MacroRequiresName.at(()).into()
            }]
        )
    }

    fn test_macro_definition(
        name: &str,
        params: impl Borrow<[&'static str]>,
        body: impl Borrow<[SemanticToken]>,
    ) {
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
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
                .act_on_eos(());
        }
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
            session.log(),
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
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let mut actions = session.semantic_actions();
            actions.emit_diag(diagnostic.clone());
            actions.did_parse_line(()).act_on_eos(());
        }
        assert_eq!(
            session.log(),
            [Event::EmitDiag {
                diag: diagnostic.into()
            }]
        )
    }

    #[test]
    fn recover_from_malformed_expr() {
        let diagnostic = Message::UnexpectedToken { token: () }.at(());
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            let mut expr = actions
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
                .act_on_eos(());
        }
        assert_eq!(
            session.log(),
            [Event::EmitDiag {
                diag: diagnostic.into()
            }]
        )
    }

    #[test]
    fn diagnose_eos_in_macro_body() {
        let mut fixture = TestFixture::<MockSpan<_>>::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
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
                .act_on_eos("eos".into());
        }
        assert_eq!(
            session.log(),
            [Event::EmitDiag {
                diag: Message::UnexpectedEof.at("eos".into()).into()
            }]
        )
    }

    #[test]
    fn diagnose_unknown_mnemonic() {
        let name = "unknown";
        let mut fixture = TestFixture::<MockSpan<_>>::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr(name.into(), name.into())
                .error()
                .unwrap()
                .did_parse_instr()
                .did_parse_line("eol".into())
                .act_on_eos("eos".into());
        }
        assert_eq!(
            session.log(),
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
        let mut fixture = TestFixture::<MockSpan<_>>::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr(name.into(), name.into())
                .error()
                .unwrap()
                .did_parse_instr()
                .did_parse_line("eol".into())
                .act_on_eos("eos".into());
        }
        assert_eq!(
            session.log(),
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
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
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
                .act_on_eos(());
        }
        assert_eq!(
            session.log(),
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
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
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
            call.did_parse_instr().did_parse_line(()).act_on_eos(());
        }
        assert_eq!(
            session.log(),
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
        let mut fixture = TestFixture::<MockSpan<_>>::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
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
                .act_on_eos("eos".into());
        }
        assert_eq!(
            session.log(),
            [Event::EmitDiag {
                diag: Message::OnlyIdentsCanBeCalled.at("literal".into()).into()
            }]
        );
    }

    #[test]
    fn diagnose_keyword_in_expr() {
        let mut fixture = TestFixture::<MockSpan<_>>::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
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
                .act_on_eos("eos".into());
        }
        assert_eq!(
            session.log(),
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
        let mut fixture = TestFixture::<MockSpan<_>>::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
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
                .act_on_eos("eos".into());
        }
        let expected = [Event::EmitFragment {
            fragment: Fragment::Immediate(
                Expr::from_atom(Atom::Name(Name::Symbol(SymbolId(0))), "f".into()),
                Width::Byte,
            ),
        }];
        assert_eq!(session.log(), expected)
    }

    #[test]
    fn act_on_known_symbol_name() {
        let mut fixture = TestFixture::<MockSpan<_>>::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
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
                .act_on_eos("eos".into());
        }
        let expected = [Event::EmitFragment {
            fragment: Fragment::Immediate(
                Expr(vec![
                    ExprOp::Atom(Atom::Name(Name::Symbol(SymbolId(0)))).with_span("f1".into()),
                    ExprOp::Atom(Atom::Name(Name::Symbol(SymbolId(0)))).with_span("f2".into()),
                    ExprOp::Binary(BinOp::Plus).with_span("plus".into()),
                ]),
                Width::Byte,
            ),
        }];
        assert_eq!(session.log(), expected)
    }

    #[test]
    fn handle_deref_const() {
        let mut fixture = TestFixture::<MockSpan<_>>::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
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
                .act_on_eos("eos".into());
        }
        let expected = [Event::EmitFragment {
            fragment: Fragment::LdInlineAddr(
                0xf0,
                Expr::from_atom(Atom::Name(Name::Symbol(SymbolId(0))), "const".into()),
            ),
        }];
        assert_eq!(session.log(), expected)
    }

    #[test]
    fn handle_param() {
        let mut fixture = TestFixture::<MockSpan<_>>::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            let mut context = actions
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
                .did_parse_line("eol".into());
        }
        let expected = [Event::DefineSymbol {
            symbol: ("label".into(), "label".into()),
            def: SymbolDef::Closure(Expr::from_atom(Atom::Param(ParamId(0)), "param2".into())),
        }];
        assert_eq!(session.log(), expected)
    }
}
