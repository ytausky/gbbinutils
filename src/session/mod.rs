use self::builder::{Backend, SymbolSource};
use self::reentrancy::ReentrancyActions;
use self::resolve::{NameTable, ResolvedName, StartScope};

use crate::analyze::macros::MacroSource;
use crate::analyze::StringSource;
use crate::diag::span::SpanSource;
use crate::diag::Diagnostics;
use crate::semantics::keywords::KEYWORDS;
use crate::semantics::Keyword;
use crate::syntax::IdentSource;

pub mod builder;
pub mod reentrancy;
pub mod resolve;

pub(crate) trait Session:
    SpanSource
    + ReentrancyActions
    + Backend<<Self as SpanSource>::Span>
    + Diagnostics<<Self as SpanSource>::Span>
    + StartScope<<Self as IdentSource>::Ident>
    + NameTable<<Self as IdentSource>::Ident, Keyword = &'static Keyword>
{
}

impl<T> Session for T where
    Self: SpanSource
        + ReentrancyActions
        + Backend<<Self as SpanSource>::Span>
        + Diagnostics<<Self as SpanSource>::Span>
        + StartScope<<Self as IdentSource>::Ident>
        + NameTable<<Self as IdentSource>::Ident, Keyword = &'static Keyword>
{
}

pub(crate) struct CompositeSession<R, N, B, D> {
    pub reentrancy: R,
    pub names: N,
    pub builder: B,
    pub diagnostics: D,
}

impl<R, N, B, D> CompositeSession<R, N, B, D>
where
    Self: ReentrancyActions,
    <Self as IdentSource>::Ident: for<'r> From<&'r str>,
    Self: NameTable<<Self as IdentSource>::Ident, Keyword = &'static Keyword>,
    Self: Backend<<Self as SpanSource>::Span>,
{
    pub fn from_components(reentrancy: R, names: N, builder: B, diagnostics: D) -> Self {
        let mut session = Self {
            reentrancy,
            names,
            builder,
            diagnostics,
        };
        for (ident, keyword) in KEYWORDS {
            session.define_name((*ident).into(), ResolvedName::Keyword(keyword))
        }
        session
    }
}

impl<R: SpanSource, N, B, D> SpanSource for CompositeSession<R, N, B, D> {
    type Span = R::Span;
}

impl<R: IdentSource, N, B, D> IdentSource for CompositeSession<R, N, B, D> {
    type Ident = R::Ident;
}

impl<R: MacroSource, N, B, D> MacroSource for CompositeSession<R, N, B, D> {
    type MacroId = R::MacroId;
}

impl<R: StringSource, N, B, D> StringSource for CompositeSession<R, N, B, D> {
    type StringRef = R::StringRef;
}

impl<R, N, B: SymbolSource, D> SymbolSource for CompositeSession<R, N, B, D> {
    type SymbolId = B::SymbolId;
}

delegate_diagnostics! {
    {R, N, B, D: Diagnostics<S>, S}, CompositeSession<R, N, B, D>, {diagnostics}, D, S
}
