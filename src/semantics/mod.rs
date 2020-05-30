use self::arg::{Arg, OperandSymbol};
use self::keywords::BuiltinMnemonic;

use crate::analyze::{Literal, StringSource, TokenSeq};
use crate::diag::span::{SpanSource, Spanned};
use crate::diag::Diagnostics;
use crate::session::builder::*;
use crate::session::reentrancy::Params;
use crate::session::Session;
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

pub(crate) struct Semantics<'a, S, T, I, R, Z> {
    pub session: S,
    pub state: T,
    pub tokens: TokenIterRef<'a, I, R, Z>,
}

type TokenIterRef<'a, I, R, S> =
    &'a mut dyn Iterator<Item = LexerOutput<I, Literal<R>, LexError, S>>;

impl<'a, S: Session, T> Semantics<'a, S, T, S::Ident, S::StringRef, S::Span> {
    fn map_state<F: FnOnce(T) -> U, U>(
        self,
        f: F,
    ) -> Semantics<'a, S, U, S::Ident, S::StringRef, S::Span> {
        Semantics {
            session: self.session,
            state: f(self.state),
            tokens: self.tokens,
        }
    }
}

type TokenStreamSemantics<'a, S> = Semantics<
    'a,
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

type InstrLineSemantics<'a, S> = Semantics<
    'a,
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

type TokenLineSemantics<'a, S> = Semantics<
    'a,
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

type BuiltinInstrSemantics<'a, S> = Semantics<
    'a,
    S,
    BuiltinInstrState<S>,
    <S as IdentSource>::Ident,
    <S as StringSource>::StringRef,
    <S as SpanSource>::Span,
>;

pub(crate) struct BuiltinInstrState<S: Session> {
    label: Option<Label<S::Ident, S::Span>>,
    mnemonic: Spanned<BuiltinMnemonic, S::Span>,
    args: BuiltinInstrArgs<S::Value, S::StringRef, S::Span>,
}

impl<S: Session> BuiltinInstrState<S> {
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

type BuiltinInstrArgs<V, R, S> = Vec<Arg<V, R, S>>;

pub(crate) type ArgSemantics<'a, S> = Semantics<
    'a,
    S,
    ExprBuilder<
        <<S as Finish>::Parent as StringSource>::StringRef,
        <<S as Finish>::Parent as SpanSource>::Span,
        BuiltinInstrState<<S as Finish>::Parent>,
    >,
    <<S as Finish>::Parent as IdentSource>::Ident,
    <<S as Finish>::Parent as StringSource>::StringRef,
    <<S as Finish>::Parent as SpanSource>::Span,
>;

pub(crate) struct ExprBuilder<R, S, P> {
    arg: Option<Arg<(), R, S>>,
    parent: P,
}

impl<R, S, P> ExprBuilder<R, S, P> {
    pub fn new(parent: P) -> Self {
        Self { arg: None, parent }
    }
}
