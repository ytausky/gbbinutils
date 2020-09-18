use self::builder::{Backend, ObjectBuilder, SymbolSource};
use self::lex::{LexItem, Literal, StringSource};
use self::macros::{MacroId, MacroSource, MacroTable, VecMacroTable};
use self::reentrancy::ReentrancyActions;
use self::resolve::*;

use crate::codebase::{BufId, BufRange, FileCodebase, FileSystem};
use crate::object::SymbolId;
use crate::semantics::keywords::KEYWORDS;
use crate::session::diagnostics::{Diagnostics, OutputForwarder};
use crate::span::{MacroDefSpans, MergeSpans, RcContextFactory, RcSpan, SpanSource, StripSpan};
use crate::syntax::{IdentFactory, IdentSource, Sigil, Token};
use crate::BuiltinSymbols;

use std::rc::Rc;

pub mod builder;
#[macro_use]
pub mod diagnostics;
pub mod lex;
pub mod macros;
pub mod reentrancy;
pub mod resolve;

pub(crate) trait Analysis:
    SpanSource
    + IdentSource
    + StringSource
    + MacroSource
    + NextToken
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
        + NextToken
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

pub(crate) trait NextToken: IdentSource + StringSource + SpanSource {
    fn next_token(&mut self) -> Option<LexItem<Self::Ident, Self::StringRef, Self::Span>>;
}

pub(crate) type Session<'a> = CompositeSession<
    FileCodebase<'a, dyn FileSystem>,
    RcContextFactory,
    VecMacroTable<Ident<String>, Literal<String>, Rc<MacroDefSpans<RcSpan<BufId, BufRange>>>>,
    BiLevelNameTable<BasicNameTable<MacroId, SymbolId>>,
    ObjectBuilder<RcSpan<BufId, BufRange>>,
    OutputForwarder<'a>,
>;

impl<'a> Session<'a> {
    pub fn new(
        codebase: FileCodebase<'a, dyn FileSystem>,
        diagnostics: OutputForwarder<'a>,
    ) -> Self {
        let mut session = Self {
            codebase,
            registry: RcContextFactory::new(),
            tokens: Vec::new(),
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

pub(crate) struct CompositeSession<C, R: SpanSource, M, N, B, D> {
    pub codebase: C,
    pub registry: R,
    tokens: Vec<
        Box<
            dyn Iterator<
                Item = LexItem<
                    <Self as IdentSource>::Ident,
                    <Self as StringSource>::StringRef,
                    R::Span,
                >,
            >,
        >,
    >,
    macros: M,
    names: N,
    pub builder: B,
    pub diagnostics: D,
}

impl<C, R: SpanSource, M, N, B, D> SpanSource for CompositeSession<C, R, M, N, B, D> {
    type Span = R::Span;
}

impl<C, R: SpanSource, M, N, B, D> IdentSource for CompositeSession<C, R, M, N, B, D> {
    type Ident = Ident<String>;
}

impl<C, R: SpanSource, M, N, B, D> StringSource for CompositeSession<C, R, M, N, B, D> {
    type StringRef = String;
}

impl<C, R: SpanSource, M, N, B: SymbolSource, D> SymbolSource
    for CompositeSession<C, R, M, N, B, D>
{
    type SymbolId = B::SymbolId;
}

impl<C, R: SpanSource, M, N, B, D> NextToken for CompositeSession<C, R, M, N, B, D> {
    fn next_token(&mut self) -> Option<LexItem<Self::Ident, Self::StringRef, Self::Span>> {
        let token = self.tokens.last_mut().unwrap().next().unwrap();
        if let Ok(Token::Sigil(Sigil::Eos)) = token.0 {
            self.tokens.pop();
        }
        Some(token)
    }
}

impl<C, R: SpanSource, M, N, B, D, S> MergeSpans<S> for CompositeSession<C, R, M, N, B, D>
where
    R: MergeSpans<S>,
    S: Clone,
{
    fn merge_spans(&mut self, left: &S, right: &S) -> S {
        self.registry.merge_spans(left, right)
    }
}

impl<C, R: SpanSource, M, N, B, D, S> StripSpan<S> for CompositeSession<C, R, M, N, B, D>
where
    R: StripSpan<S>,
    S: Clone,
{
    type Stripped = R::Stripped;

    fn strip_span(&mut self, span: &S) -> Self::Stripped {
        self.registry.strip_span(span)
    }
}

#[cfg(test)]
pub mod mock {
    use super::*;

    use super::builder::mock::{MockBackend, MockSymbolId, SerialIdAllocator};
    use super::macros::mock::{MockMacroId, MockMacroTable};
    use super::reentrancy::MockCodebase;

    use crate::codebase::CodebaseError;
    use crate::log::Log;
    use crate::session::diagnostics::{MockDiagnostics, TestDiagnosticsListener};

    pub(crate) type MockSession<T, S> = CompositeSession<
        MockCodebase<T, S>,
        MockDiagnostics<T, S>,
        MockMacroTable<T>,
        MockNameTable<BiLevelNameTable<BasicNameTable<MockMacroId, MockSymbolId>>, T>,
        MockBackend<SerialIdAllocator<MockSymbolId>, T>,
        MockDiagnostics<T, S>,
    >;

    impl<T, S: Clone> MockSession<T, S> {
        pub fn new(log: Log<T>) -> Self {
            let mut names = BiLevelNameTable::new();
            for (ident, keyword) in KEYWORDS {
                names.define_name((*ident).into(), ResolvedName::Keyword(keyword))
            }
            CompositeSession {
                codebase: MockCodebase::with_log(log.clone()),
                registry: MockDiagnostics::new(log.clone()),
                tokens: Vec::new(),
                macros: MockMacroTable::new(log.clone()),
                names: MockNameTable::new(names, log.clone()),
                builder: MockBackend::new(SerialIdAllocator::new(MockSymbolId), log.clone()),
                diagnostics: MockDiagnostics::new(log),
            }
        }

        pub fn fail(&mut self, error: CodebaseError) {
            self.codebase.fail(error)
        }
    }

    pub(crate) type StandaloneBackend<S> = CompositeSession<
        (),
        TestDiagnosticsListener<S>,
        (),
        (),
        ObjectBuilder<S>,
        TestDiagnosticsListener<S>,
    >;

    impl<S: Clone> StandaloneBackend<S> {
        pub fn new() -> Self {
            CompositeSession {
                codebase: (),
                registry: TestDiagnosticsListener::new(),
                tokens: Vec::new(),
                macros: (),
                names: (),
                builder: ObjectBuilder::new(),
                diagnostics: TestDiagnosticsListener::new(),
            }
        }
    }
}
