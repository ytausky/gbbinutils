use self::builder::{Backend, ObjectBuilder, SymbolSource};
use self::lex::{CodebaseAnalyzer, Literal, StringSource, Tokenizer};
use self::macros::{MacroId, MacroSource, MacroTable, VecMacroTable};
use self::reentrancy::{ReentrancyActions, SourceComponents};
use self::resolve::*;
use self::strings::FakeStringInterner;

use crate::codebase::{BufId, BufRange, FileCodebase, FileSystem};
use crate::diag::{CompositeDiagnosticsSystem, Diagnostics, OutputForwarder};
use crate::object::SymbolId;
use crate::semantics::keywords::KEYWORDS;
use crate::span::{MacroDefSpans, RcContextFactory, RcSpan, SpanSource};
use crate::syntax::parser::DefaultParserFactory;
use crate::syntax::{IdentFactory, IdentSource};
use crate::BuiltinSymbols;

use std::rc::Rc;

pub mod builder;
pub mod lex;
pub mod macros;
pub mod reentrancy;
pub mod resolve;
pub mod strings;

pub(crate) trait Analysis:
    SpanSource
    + IdentSource
    + StringSource
    + MacroSource
    + ReentrancyActions<<Self as StringSource>::StringRef>
    + Backend<<Self as SpanSource>::Span>
    + Diagnostics<<Self as SpanSource>::Span>
    + StartScope<<Self as IdentSource>::Ident>
    + NameTable<<Self as IdentSource>::Ident>
    + MacroTable<
        <Self as IdentSource>::Ident,
        Literal<<Self as StringSource>::StringRef>,
        <Self as SpanSource>::Span,
    >
{
}

impl<T> Analysis for T where
    Self: SpanSource
        + IdentSource
        + StringSource
        + MacroSource
        + ReentrancyActions<<Self as StringSource>::StringRef>
        + Backend<<Self as SpanSource>::Span>
        + Diagnostics<<Self as SpanSource>::Span>
        + StartScope<<Self as IdentSource>::Ident>
        + NameTable<<Self as IdentSource>::Ident>
        + MacroTable<
            <Self as IdentSource>::Ident,
            Literal<<Self as StringSource>::StringRef>,
            <Self as SpanSource>::Span,
        >
{
}

pub(crate) type Session<'a, 'b> = CompositeSession<
    SourceComponents<
        CodebaseAnalyzer<'b, Tokenizer<&'b FileCodebase<'a, dyn FileSystem>>>,
        DefaultParserFactory,
        FakeStringInterner,
    >,
    VecMacroTable<Ident<String>, Literal<String>, Rc<MacroDefSpans<RcSpan<BufId, BufRange>>>>,
    BiLevelNameTable<BasicNameTable<MacroId, SymbolId>>,
    ObjectBuilder<RcSpan<BufId, BufRange>>,
    CompositeDiagnosticsSystem<RcContextFactory, OutputForwarder<'a, 'b>>,
>;

impl<'a, 'b> Session<'a, 'b> {
    pub fn new(
        codebase: CodebaseAnalyzer<'b, Tokenizer<&'b FileCodebase<'a, dyn FileSystem>>>,
        diagnostics: CompositeDiagnosticsSystem<RcContextFactory, OutputForwarder<'a, 'b>>,
    ) -> Self {
        let mut session = Self {
            reentrancy: SourceComponents {
                codebase,
                parser_factory: DefaultParserFactory,
                interner: FakeStringInterner,
            },
            macros: VecMacroTable::new(),
            names: BiLevelNameTable::new(),
            builder: ObjectBuilder::new(),
            diagnostics,
        };
        for (string, name) in session.builder.builtin_symbols() {
            session.names.define_name(
                DefaultIdentFactory.mk_ident(string),
                ResolvedName::Symbol(*name),
            )
        }
        for (ident, keyword) in KEYWORDS {
            session.define_name((*ident).into(), ResolvedName::Keyword(keyword))
        }
        session
    }
}

pub(crate) struct CompositeSession<R, M, N, B, D> {
    reentrancy: R,
    macros: M,
    names: N,
    pub builder: B,
    pub diagnostics: D,
}

#[cfg(test)]
impl<R, M, N, B, D> CompositeSession<R, M, N, B, D>
where
    Self: IdentSource + SpanSource,
    <Self as IdentSource>::Ident: for<'r> From<&'r str>,
    Self: NameTable<<Self as IdentSource>::Ident>,
    Self: Backend<<Self as SpanSource>::Span>,
{
    pub fn from_components(reentrancy: R, macros: M, names: N, builder: B, diagnostics: D) -> Self {
        let mut session = Self {
            reentrancy,
            macros,
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

impl<R: SpanSource, M, N, B, D> SpanSource for CompositeSession<R, M, N, B, D> {
    type Span = R::Span;
}

impl<R: IdentSource, M, N, B, D> IdentSource for CompositeSession<R, M, N, B, D> {
    type Ident = R::Ident;
}

impl<R: StringSource, M, N, B, D> StringSource for CompositeSession<R, M, N, B, D> {
    type StringRef = R::StringRef;
}

impl<R, M, N, B: SymbolSource, D> SymbolSource for CompositeSession<R, M, N, B, D> {
    type SymbolId = B::SymbolId;
}

delegate_diagnostics! {
    {R, M, N, B, D: Diagnostics<S>, S}, CompositeSession<R, M, N, B, D>, {diagnostics}, D, S
}

#[cfg(test)]
pub mod mock {
    use super::*;

    use super::builder::mock::{MockBackend, MockSymbolId, SerialIdAllocator};
    use super::macros::mock::{MockMacroId, MockMacroTable};
    use super::reentrancy::MockSourceComponents;

    use crate::codebase::CodebaseError;
    use crate::diag::{MockDiagnostics, TestDiagnosticsListener};
    use crate::log::Log;

    pub(crate) type MockSession<T, S> = CompositeSession<
        MockSourceComponents<T, S>,
        MockMacroTable<T>,
        MockNameTable<BasicNameTable<MockMacroId, MockSymbolId>, T>,
        MockBackend<SerialIdAllocator<MockSymbolId>, T>,
        MockDiagnostics<T, S>,
    >;

    impl<T, S> MockSession<T, S> {
        pub fn new(log: Log<T>) -> Self {
            CompositeSession {
                reentrancy: MockSourceComponents::with_log(log.clone()),
                macros: MockMacroTable::new(log.clone()),
                names: MockNameTable::new(BasicNameTable::default(), log.clone()),
                builder: MockBackend::new(SerialIdAllocator::new(MockSymbolId), log.clone()),
                diagnostics: MockDiagnostics::new(log),
            }
        }

        pub fn fail(&mut self, error: CodebaseError) {
            self.reentrancy.fail(error)
        }
    }

    pub(crate) type StandaloneBackend<S> =
        CompositeSession<(), (), (), ObjectBuilder<S>, TestDiagnosticsListener<S>>;

    impl<S> StandaloneBackend<S> {
        pub fn new() -> Self {
            CompositeSession {
                reentrancy: (),
                macros: (),
                names: (),
                builder: ObjectBuilder::new(),
                diagnostics: TestDiagnosticsListener::new(),
            }
        }
    }
}
