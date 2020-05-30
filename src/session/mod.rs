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

pub(crate) struct CompositeSession<R, N, B> {
    pub reentrancy: R,
    pub names: N,
    pub builder: B,
}

impl<R, N, B> CompositeSession<R, N, B>
where
    Self: ReentrancyActions,
    <Self as IdentSource>::Ident: for<'r> From<&'r str>,
    Self: NameTable<<Self as IdentSource>::Ident, Keyword = &'static Keyword>,
    Self: Backend<<Self as SpanSource>::Span>,
{
    pub fn from_components(reentrancy: R, names: N, builder: B) -> Self {
        let mut session = Self {
            reentrancy,
            names,
            builder,
        };
        for (ident, keyword) in KEYWORDS {
            session.define_name((*ident).into(), ResolvedName::Keyword(keyword))
        }
        session
    }
}

impl<R: SpanSource, N, B> SpanSource for CompositeSession<R, N, B> {
    type Span = R::Span;
}

impl<R: IdentSource, N, B> IdentSource for CompositeSession<R, N, B> {
    type Ident = R::Ident;
}

impl<R: MacroSource, N, B> MacroSource for CompositeSession<R, N, B> {
    type MacroId = R::MacroId;
}

impl<R: StringSource, N, B> StringSource for CompositeSession<R, N, B> {
    type StringRef = R::StringRef;
}

impl<R, N, B: SymbolSource> SymbolSource for CompositeSession<R, N, B> {
    type SymbolId = B::SymbolId;
}
