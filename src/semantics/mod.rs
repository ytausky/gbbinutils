use self::arg::{Arg, OperandSymbol, ParsedArg};
use self::keywords::BuiltinMnemonic;

use crate::session::builder::*;
use crate::session::diagnostics::Diagnostics;
use crate::session::lex::{StringSource, TokenSeq};
use crate::session::reentrancy::Params;
use crate::session::Analysis;
use crate::span::{SpanSource, Spanned};
use crate::syntax::actions::{LexerOutput, LineRule};

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

type TokenStreamSemantics<'a, S> =
    Semantics<'a, S, TokenStreamState<<S as StringSource>::StringRef, <S as SpanSource>::Span>>;

#[derive(Debug, PartialEq)]
pub struct TokenStreamState<R, S> {
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

impl<R, S> MacroDefState<R, S> {
    fn new(label: Option<Label<R, S>>) -> Self {
        Self {
            label,
            tokens: Vec::new(),
        }
    }
}

type BuiltinInstrSemantics<'a, S> = Semantics<'a, S, BuiltinInstrState<S>>;

pub(crate) struct BuiltinInstrState<S: Analysis> {
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

pub(crate) type ArgSemantics<'a, S> = Semantics<
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
