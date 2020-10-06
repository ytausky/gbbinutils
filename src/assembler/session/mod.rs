use self::builder::{Backend, ObjectBuilder, SymbolSource};
use self::lex::{LexItem, Literal, StringSource};
use self::macros::{MacroId, MacroSource, MacroTable, VecMacroTable};
use self::reentrancy::ReentrancyActions;
use self::resolve::*;

use crate::assembler::semantics::keywords::KEYWORDS;
use crate::assembler::syntax::{Sigil, Token};
use crate::codebase::{BufId, FileCodebase, FileSystem};
use crate::diagnostics::{CompactDiag, Diagnostics, DiagnosticsContext, EmitDiag, OutputForwarder};
use crate::object::SymbolId;
use crate::span::*;

use std::collections::HashMap;
use std::rc::Rc;

pub mod builder;
pub mod lex;
pub mod macros;
pub mod reentrancy;
pub mod resolve;

pub(crate) trait Analysis:
    SpanSource
    + StringSource
    + MacroSource
    + NextToken
    + Interner
    + ReentrancyActions<<Self as StringSource>::StringRef, <Self as SpanSource>::Span>
    + Backend<<Self as SpanSource>::Span>
    + Diagnostics<<Self as SpanSource>::Span>
    + StartScope
    + NameTable<<Self as StringSource>::StringRef>
    + MacroTable<
        <Self as StringSource>::StringRef,
        Literal<<Self as StringSource>::StringRef>,
        <Self as SpanSource>::Span,
    >
{
}

impl<T> Analysis for T where
    Self: SpanSource
        + StringSource
        + MacroSource
        + NextToken
        + Interner
        + ReentrancyActions<<Self as StringSource>::StringRef, <Self as SpanSource>::Span>
        + Backend<<Self as SpanSource>::Span>
        + Diagnostics<<Self as SpanSource>::Span>
        + StartScope
        + NameTable<<Self as StringSource>::StringRef>
        + MacroTable<
            <Self as StringSource>::StringRef,
            Literal<<Self as StringSource>::StringRef>,
            <Self as SpanSource>::Span,
        >
{
}

pub(crate) trait NextToken: StringSource + SpanSource {
    fn next_token(&mut self) -> Option<LexItem<Self::StringRef, Self::Span>>;
}

pub(crate) type Session<'a> = CompositeSession<
    FileCodebase<'a, dyn FileSystem>,
    RcContextFactory<BufId>,
    HashInterner,
    VecMacroTable<
        Rc<MacroDefMetadata<Span<RcFileInclusion<BufId>, RcMacroExpansion<BufId>>>>,
        StringId,
    >,
    BiLevelNameTable<MacroId, SymbolId, StringId>,
    ObjectBuilder<Span<RcFileInclusion<BufId>, RcMacroExpansion<BufId>>>,
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
            interner: HashInterner::new(),
            tokens: Vec::new(),
            macros: VecMacroTable::new(),
            names: BiLevelNameTable::new(),
            builder: ObjectBuilder::new(),
            diagnostics,
        };
        for (string, name) in crate::eval::BUILTIN_SYMBOLS {
            let string = session.interner.intern(string);
            session.define_name_with_visibility(
                string,
                Visibility::Global,
                ResolvedName::Symbol(*name),
            )
        }
        for (ident, keyword) in KEYWORDS {
            let string = session.interner.intern(ident);
            session.define_name_with_visibility(
                string,
                Visibility::Global,
                ResolvedName::Keyword(keyword),
            )
        }
        session
    }
}

pub(crate) trait TokenStream<R: SpanSource, I: StringSource> {
    fn next_token(
        &mut self,
        registry: &mut R,
        interner: &mut I,
    ) -> Option<LexItem<I::StringRef, R::Span>>;
}

pub(crate) struct CompositeSession<C, R: SpanSource, I: StringSource, M, N, B, D> {
    pub codebase: C,
    pub registry: R,
    interner: I,
    tokens: Vec<Box<dyn TokenStream<R, I>>>,
    macros: M,
    names: N,
    pub builder: B,
    pub diagnostics: D,
}

impl<C, R: SpanSource, I: StringSource, M, N, B, D> SpanSource
    for CompositeSession<C, R, I, M, N, B, D>
{
    type Span = R::Span;
}

impl<C, R: SpanSource, I: StringSource, M, N, B, D> StringSource
    for CompositeSession<C, R, I, M, N, B, D>
{
    type StringRef = I::StringRef;
}

impl<C, R: SpanSource, I: StringSource, M, N, B: SymbolSource, D> SymbolSource
    for CompositeSession<C, R, I, M, N, B, D>
{
    type SymbolId = B::SymbolId;
}

impl<C, R: SpanSource, I: StringSource, M, N, B, D> NextToken
    for CompositeSession<C, R, I, M, N, B, D>
{
    fn next_token(&mut self) -> Option<LexItem<Self::StringRef, Self::Span>> {
        let token = self
            .tokens
            .last_mut()
            .unwrap()
            .next_token(&mut self.registry, &mut self.interner)
            .unwrap();
        if let Ok(Token::Sigil(Sigil::Eos)) = token.0 {
            self.tokens.pop();
        }
        Some(token)
    }
}

impl<C, R: SpanSource, I: StringSource, M, N, B, D, S, Stripped> EmitDiag<S, Stripped>
    for CompositeSession<C, R, I, M, N, B, D>
where
    for<'a> DiagnosticsContext<'a, C, R, D>: EmitDiag<S, Stripped>,
{
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<S, Stripped>>) {
        self.diagnostics().emit_diag(diag)
    }
}

impl<C, R: SpanSource, I: StringSource, M, N, B, D> CompositeSession<C, R, I, M, N, B, D> {
    fn diagnostics(&mut self) -> DiagnosticsContext<C, R, D> {
        DiagnosticsContext {
            codebase: &mut self.codebase,
            registry: &mut self.registry,
            diagnostics: &mut self.diagnostics,
        }
    }
}

impl<C, R: SpanSource, I: StringSource, M, N, B, D, S> MergeSpans<S>
    for CompositeSession<C, R, I, M, N, B, D>
where
    R: MergeSpans<S>,
    S: Clone,
{
    fn merge_spans(&mut self, left: &S, right: &S) -> S {
        self.registry.merge_spans(left, right)
    }
}

impl<C, R: SpanSource, I: StringSource, M, N, B, D, S> StripSpan<S>
    for CompositeSession<C, R, I, M, N, B, D>
where
    R: StripSpan<S>,
    S: Clone,
{
    type Stripped = R::Stripped;

    fn strip_span(&mut self, span: &S) -> Self::Stripped {
        self.registry.strip_span(span)
    }
}

impl<C, R: SpanSource, I: Interner, M, N, B, D> Interner for CompositeSession<C, R, I, M, N, B, D> {
    fn intern(&mut self, string: &str) -> Self::StringRef {
        self.interner.intern(string)
    }

    fn get_string<'a>(&'a self, id: &'a Self::StringRef) -> &str {
        self.interner.get_string(id)
    }
}

pub(crate) trait Interner: StringSource {
    fn intern(&mut self, string: &str) -> Self::StringRef;
    fn get_string<'a>(&'a self, id: &'a Self::StringRef) -> &str;
}

pub struct MockInterner;

impl StringSource for MockInterner {
    type StringRef = String;
}

impl Interner for MockInterner {
    fn intern(&mut self, string: &str) -> Self::StringRef {
        string.to_owned()
    }

    fn get_string<'a>(&'a self, id: &'a Self::StringRef) -> &str {
        id
    }
}

pub(crate) struct HashInterner {
    map: HashMap<String, StringId>,
    strings: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct StringId(usize);

impl HashInterner {
    fn new() -> Self {
        HashInterner {
            map: HashMap::new(),
            strings: Vec::new(),
        }
    }
}

impl StringSource for HashInterner {
    type StringRef = StringId;
}

impl Interner for HashInterner {
    fn intern(&mut self, string: &str) -> Self::StringRef {
        let strings = &mut self.strings;
        let id = self
            .map
            .entry(string.to_owned())
            .or_insert_with(|| StringId(strings.len()));
        if id.0 == strings.len() {
            strings.push(string.to_owned())
        }
        *id
    }

    fn get_string<'a>(&'a self, id: &'a Self::StringRef) -> &str {
        &self.strings[id.0]
    }
}

#[cfg(test)]
pub mod mock {
    use super::*;

    use super::builder::mock::{MockBackend, MockSymbolId, SerialIdAllocator};
    use super::macros::mock::{MockMacroId, MockMacroTable};
    use super::reentrancy::MockCodebase;

    use crate::codebase::CodebaseError;
    use crate::diagnostics::{MockDiagnostics, TestDiagnosticsListener};
    use crate::log::Log;

    pub(crate) type MockSession<T, S> = CompositeSession<
        MockCodebase<T, S>,
        MockDiagnostics<T, S>,
        MockInterner,
        MockMacroTable<T>,
        MockNameTable<BiLevelNameTable<MockMacroId, MockSymbolId, String>, T>,
        MockBackend<SerialIdAllocator<MockSymbolId>, T>,
        MockDiagnostics<T, S>,
    >;

    impl<T, S: Clone> MockSession<T, S> {
        pub fn new(log: Log<T>) -> Self {
            let mut names = BiLevelNameTable::new();
            for (ident, keyword) in KEYWORDS {
                names
                    .global
                    .insert((*ident).into(), ResolvedName::Keyword(keyword));
            }
            CompositeSession {
                codebase: MockCodebase::with_log(log.clone()),
                registry: MockDiagnostics::new(log.clone()),
                interner: MockInterner,
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
        MockInterner,
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
                interner: MockInterner,
                tokens: Vec::new(),
                macros: (),
                names: (),
                builder: ObjectBuilder::new(),
                diagnostics: TestDiagnosticsListener::new(),
            }
        }
    }
}
