use self::arg::{Arg, OperandSymbol, ParsedArg};
use self::keywords::BuiltinMnemonic;

use crate::session::builder::*;
use crate::session::diagnostics::Diagnostics;
use crate::session::lex::{StringSource, TokenSeq};
use crate::session::reentrancy::Params;
use crate::session::Analysis;
use crate::span::{SpanSource, Spanned};
use crate::syntax::actions::{LexerOutput, LineRule};
use crate::syntax::IdentSource;

macro_rules! set_state {
    ($session:expr, $state:expr) => {
        $crate::semantics::Semantics {
            session: $session.session,
            state: $state,
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

pub(crate) struct Semantics<'a, S, T> {
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

type TokenStreamSemantics<'a, S> = Semantics<
    'a,
    S,
    TokenStreamState<
        <S as IdentSource>::Ident,
        <S as StringSource>::StringRef,
        <S as SpanSource>::Span,
    >,
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

type InstrLineSemantics<'a, S> =
    Semantics<'a, S, InstrLineState<<S as IdentSource>::Ident, <S as SpanSource>::Span>>;

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

type BuiltinInstrSemantics<'a, S> = Semantics<'a, S, BuiltinInstrState<S>>;

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

pub(crate) type ArgSemantics<'a, S> = Semantics<
    'a,
    S,
    ExprBuilder<
        <S as IdentSource>::Ident,
        <S as StringSource>::StringRef,
        <S as SpanSource>::Span,
        BuiltinInstrState<S>,
    >,
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
