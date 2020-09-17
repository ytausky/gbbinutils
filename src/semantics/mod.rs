use self::arg::{Arg, OperandSymbol, ParsedArg};
use self::keywords::BuiltinMnemonic;

use crate::session::lex::{Literal, StringSource, TokenSeq};
use crate::diag::Diagnostics;
use crate::session::builder::*;
use crate::session::reentrancy::Params;
use crate::session::Analysis;
use crate::span::{SpanSource, Spanned};
use crate::syntax::actions::{LexerOutput, LineRule};
use crate::syntax::{IdentSource, LexError};

macro_rules! set_state {
    ($session:expr, $state:expr) => {
        $crate::semantics::Semantics {
            session: $session.session,
            state: $state,
            tokens: $session.tokens,
        }
    };
}

mod actions;
mod arg;
pub mod keywords;

#[derive(Clone, Debug, PartialEq)]
pub enum Keyword {
    BuiltinMnemonic(BuiltinMnemonic),
    Operand(OperandSymbol),
}

pub(crate) struct Semantics<'a, 'b, S, T, I, R, Z> {
    pub session: &'a mut S,
    pub state: T,
    pub tokens: TokenIterRef<'b, I, R, Z>,
}

type TokenIterRef<'a, I, R, S> =
    &'a mut dyn Iterator<Item = LexerOutput<I, Literal<R>, LexError, S>>;

impl<'a, 'b, S: Analysis, T> Semantics<'a, 'b, S, T, S::Ident, S::StringRef, S::Span> {
    fn map_state<F: FnOnce(T) -> U, U>(
        self,
        f: F,
    ) -> Semantics<'a, 'b, S, U, S::Ident, S::StringRef, S::Span> {
        Semantics {
            session: self.session,
            state: f(self.state),
            tokens: self.tokens,
        }
    }
}

type TokenStreamSemantics<'a, 'b, S> = Semantics<
    'a,
    'b,
    S,
    TokenStreamState<
        <S as IdentSource>::Ident,
        <S as StringSource>::StringRef,
        <S as SpanSource>::Span,
    >,
    <S as IdentSource>::Ident,
    <S as StringSource>::StringRef,
    <S as SpanSource>::Span,
>;

#[derive(Debug, PartialEq)]
pub struct TokenStreamState<I, R, S> {
    mode: LineRule<InstrLineState<I, S>, TokenLineState<I, R, S>>,
}

impl<I, R, S> TokenStreamState<I, R, S> {
    pub fn new() -> Self {
        Self {
            mode: LineRule::InstrLine(InstrLineState::new()),
        }
    }
}

type InstrLineSemantics<'a, 'b, S> = Semantics<
    'a,
    'b,
    S,
    InstrLineState<<S as IdentSource>::Ident, <S as SpanSource>::Span>,
    <S as IdentSource>::Ident,
    <S as StringSource>::StringRef,
    <S as SpanSource>::Span,
>;

#[derive(Debug, PartialEq)]
pub struct InstrLineState<I, S> {
    label: Option<Label<I, S>>,
}

impl<I, S> InstrLineState<I, S> {
    fn new() -> Self {
        Self { label: None }
    }
}

type Label<I, S> = ((I, S), Params<I, S>);

type TokenLineSemantics<'a, 'b, S> = Semantics<
    'a,
    'b,
    S,
    TokenLineState<
        <S as IdentSource>::Ident,
        <S as StringSource>::StringRef,
        <S as SpanSource>::Span,
    >,
    <S as IdentSource>::Ident,
    <S as StringSource>::StringRef,
    <S as SpanSource>::Span,
>;

#[derive(Debug, PartialEq)]
pub struct TokenLineState<I, R, S> {
    context: TokenContext<I, R, S>,
}

#[derive(Debug, PartialEq)]
pub enum TokenContext<I, R, S> {
    FalseIf,
    MacroDef(MacroDefState<I, R, S>),
}

#[derive(Debug, PartialEq)]
pub struct MacroDefState<I, R, S> {
    label: Option<Label<I, S>>,
    tokens: TokenSeq<I, R, S>,
}

impl<I, R, S> MacroDefState<I, R, S> {
    fn new(label: Option<Label<I, S>>) -> Self {
        Self {
            label,
            tokens: (Vec::new(), Vec::new()),
        }
    }
}

type BuiltinInstrSemantics<'a, 'b, S> = Semantics<
    'a,
    'b,
    S,
    BuiltinInstrState<S>,
    <S as IdentSource>::Ident,
    <S as StringSource>::StringRef,
    <S as SpanSource>::Span,
>;

pub(crate) struct BuiltinInstrState<S: Analysis> {
    label: Option<Label<S::Ident, S::Span>>,
    mnemonic: Spanned<BuiltinMnemonic, S::Span>,
    args: BuiltinInstrArgs<S::Ident, S::StringRef, S::Span>,
}

impl<S: Analysis> BuiltinInstrState<S> {
    fn new(
        label: Option<Label<S::Ident, S::Span>>,
        mnemonic: Spanned<BuiltinMnemonic, S::Span>,
    ) -> Self {
        Self {
            label,
            mnemonic,
            args: Vec::new(),
        }
    }
}

type BuiltinInstrArgs<N, R, S> = Vec<ParsedArg<N, R, S>>;

pub(crate) type ArgSemantics<'a, 'b, S> = Semantics<
    'a,
    'b,
    S,
    ExprBuilder<
        <S as IdentSource>::Ident,
        <S as StringSource>::StringRef,
        <S as SpanSource>::Span,
        BuiltinInstrState<S>,
    >,
    <S as IdentSource>::Ident,
    <S as StringSource>::StringRef,
    <S as SpanSource>::Span,
>;

pub(crate) struct ExprBuilder<I, R, S, P> {
    arg: Option<ParsedArg<I, R, S>>,
    parent: P,
}

impl<I, R, S, P> ExprBuilder<I, R, S, P> {
    pub fn new(parent: P) -> Self {
        Self { arg: None, parent }
    }
}
