use super::keywords::{BuiltinMnemonic, OperandSymbol};

use crate::assembler::session::builder::*;
use crate::assembler::session::lex::{StringSource, TokenSeq};
use crate::assembler::session::macros::MacroSource;
use crate::assembler::session::reentrancy::Params;
use crate::assembler::session::resolve::{NameTable, ResolvedName, Visibility};
use crate::assembler::session::{Analysis, Interner};
use crate::assembler::syntax::actions::{LexerOutput, LineRule};
use crate::diagnostics::Diagnostics;
use crate::expr::Expr;
use crate::span::{SpanSource, Spanned};

macro_rules! set_state {
    ($session:expr, $state:expr) => {
        $crate::assembler::semantics::Semantics {
            session: $session.session,
            state: $state,
        }
    };
}

mod actions;

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
            tokens: (Vec::new(), Vec::new()),
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

enum ParsedArg<R, S> {
    Bare(Expr<R, S>),
    Parenthesized(Expr<R, S>, S),
    String(R, S),
    Error,
}

enum Arg<N, R, S> {
    Bare(BareArg<N, S>),
    Deref(BareArg<N, S>, S),
    String(R, S),
    Error,
}

#[derive(Clone)]
enum BareArg<N, S> {
    Const(Expr<N, S>),
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

trait DefineName<R, M, S> {
    fn define_name(&mut self, name: R, entry: ResolvedName<M, S>);
}

impl<S> DefineName<S::StringRef, S::MacroId, S::SymbolId> for S
where
    S: Interner + NameTable<<S as StringSource>::StringRef> + MacroSource + SymbolSource,
{
    fn define_name(&mut self, name: S::StringRef, entry: ResolvedName<S::MacroId, S::SymbolId>) {
        let visibility = self.name_visibility(&name);
        self.define_name_with_visibility(name, visibility, entry)
    }
}

trait ResolveName<R>: MacroSource + SymbolSource {
    fn resolve_name(&mut self, name: &R) -> Option<ResolvedName<Self::MacroId, Self::SymbolId>>;
}

impl<S> ResolveName<S::StringRef> for S
where
    S: Interner + NameTable<<S as StringSource>::StringRef>,
{
    fn resolve_name(
        &mut self,
        name: &S::StringRef,
    ) -> Option<ResolvedName<Self::MacroId, Self::SymbolId>> {
        let visibility = self.name_visibility(name);
        self.resolve_name_with_visibility(name, visibility)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::assembler::session::mock::{MockSession, TestOperation};
    use crate::log::Log;

    #[test]
    fn ident_with_underscore_prefix_is_local() {
        let session = MockSession::<TestOperation<()>, ()>::new(Log::default());
        assert_eq!(
            session.name_visibility(&"_loop".to_owned()),
            Visibility::Local
        )
    }

    #[test]
    fn ident_without_underscore_prefix_is_global() {
        let session = MockSession::<TestOperation<()>, ()>::new(Log::default());
        assert_eq!(
            session.name_visibility(&"start".to_owned()),
            Visibility::Global
        )
    }
}
