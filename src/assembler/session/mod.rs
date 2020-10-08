use self::builder::ObjectBuilder;
use self::macros::{MacroArgs, MacroTable, VecMacroTable};
use self::resolve::*;

use super::keywords::KEYWORDS;
use super::semantics::Keyword;
use super::syntax::{LexItem, Literal, SemanticToken, Sigil, Token};

use crate::codebase::{BufId, CodebaseError, FileCodebase, FileSystem};
use crate::diagnostics::{CompactDiag, Diagnostics, DiagnosticsContext, EmitDiag, OutputForwarder};
use crate::expr::Expr;
use crate::object::{Fragment, SymbolId};
use crate::span::*;

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

mod builder;
mod lex;
mod macros;
mod reentrancy;
mod resolve;

pub(super) trait Analysis:
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

pub(super) trait ReentrancyActions<R, S> {
    fn analyze_file(&mut self, path: R, from: Option<S>) -> Result<(), CodebaseError>;
}

pub(crate) trait Backend<S: Clone>: AllocSymbol<S> {
    fn define_symbol(&mut self, name: Self::SymbolId, span: S, expr: Expr<Self::SymbolId, S>);
    fn emit_fragment(&mut self, fragment: Fragment<Expr<Self::SymbolId, S>>);
    fn is_non_zero(&mut self, value: Expr<Self::SymbolId, S>) -> Option<bool>;
    fn set_origin(&mut self, origin: Expr<Self::SymbolId, S>);
    fn start_section(&mut self, name: Self::SymbolId, span: S);
}

pub trait AllocSymbol<S: Clone>: SymbolSource {
    fn alloc_symbol(&mut self, span: S) -> Self::SymbolId;
}

pub(super) trait NameTable<I>: MacroSource + SymbolSource {
    fn resolve_name_with_visibility(
        &mut self,
        ident: &I,
        visibility: Visibility,
    ) -> Option<ResolvedName<Self::MacroId, Self::SymbolId>>;

    fn define_name_with_visibility(
        &mut self,
        ident: I,
        visibility: Visibility,
        entry: ResolvedName<Self::MacroId, Self::SymbolId>,
    );
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacroId(pub usize);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Visibility {
    Global,
    Local,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ResolvedName<MacroId, SymbolId> {
    Keyword(&'static Keyword),
    Macro(MacroId),
    Symbol(SymbolId),
}

pub trait StringSource {
    type StringRef: Clone + Debug + Eq + Hash;
}

pub trait SymbolSource {
    type SymbolId: Clone;
}

pub(crate) trait MacroSource {
    type MacroId: Clone;
}

pub(super) type Session<'a> = CompositeSession<
    FileCodebase<'a, dyn FileSystem>,
    RcContextFactory<BufId>,
    HashInterner,
    OutputForwarder<'a>,
    (),
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
            log: (),
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

pub(crate) struct CompositeSession<C, R: SpanSystem<BufId>, I: StringSource, D, L> {
    pub codebase: C,
    pub registry: R,
    interner: I,
    tokens: Vec<Box<dyn TokenStream<R, I>>>,
    macros: VecMacroTable<R::MacroDefMetadataId, I::StringRef>,
    names: BiLevelNameTable<MacroId, SymbolId, I::StringRef>,
    pub builder: ObjectBuilder<R::Span>,
    pub diagnostics: D,
    #[allow(dead_code)]
    log: L,
}

impl<C, R: SpanSystem<BufId>, I: StringSource, D, L> SpanSource
    for CompositeSession<C, R, I, D, L>
{
    type Span = R::Span;
}

impl<C, R: SpanSystem<BufId>, I: StringSource, D, L> StringSource
    for CompositeSession<C, R, I, D, L>
{
    type StringRef = I::StringRef;
}

impl<C, R: SpanSystem<BufId>, I: StringSource, D, L> SymbolSource
    for CompositeSession<C, R, I, D, L>
{
    type SymbolId = SymbolId;
}

impl<C, R: SpanSystem<BufId>, I: StringSource, D, L> NextToken for CompositeSession<C, R, I, D, L> {
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

impl<C, R: SpanSystem<BufId>, I: StringSource, D, L> EmitDiag<R::Span, R::Stripped>
    for CompositeSession<C, R, I, D, L>
where
    Self: SymbolSource + MacroSource,
    Self: Log<
        <Self as SymbolSource>::SymbolId,
        <Self as MacroSource>::MacroId,
        I::StringRef,
        R::Span,
        R::Stripped,
    >,
    for<'a> DiagnosticsContext<'a, C, R, D>: EmitDiag<R::Span, R::Stripped>,
    R::Stripped: Clone,
{
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<R::Span, R::Stripped>>) {
        let diag = diag.into();
        self.log(|| Event::EmitDiag { diag: diag.clone() });
        self.diagnostics().emit_diag(diag)
    }
}

impl<C, R: SpanSystem<BufId>, I: StringSource, D, L> CompositeSession<C, R, I, D, L> {
    fn diagnostics(&mut self) -> DiagnosticsContext<C, R, D> {
        DiagnosticsContext {
            codebase: &mut self.codebase,
            registry: &mut self.registry,
            diagnostics: &mut self.diagnostics,
        }
    }
}

impl<C, R: SpanSystem<BufId>, I: StringSource, D, L> MergeSpans<R::Span>
    for CompositeSession<C, R, I, D, L>
{
    fn merge_spans(&mut self, left: &R::Span, right: &R::Span) -> R::Span {
        self.registry.merge_spans(left, right)
    }
}

impl<C, R: SpanSystem<BufId>, I: StringSource, D, L> StripSpan<R::Span>
    for CompositeSession<C, R, I, D, L>
{
    type Stripped = R::Stripped;

    fn strip_span(&mut self, span: &R::Span) -> Self::Stripped {
        self.registry.strip_span(span)
    }
}

impl<C, R: SpanSystem<BufId>, I: Interner, D, L> Interner for CompositeSession<C, R, I, D, L> {
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
impl<C, R: SpanSystem<BufId>, I: Interner, D, BB, MM, S, T>
    CompositeSession<C, R, I, D, Vec<Event<BB, MM, I::StringRef, S, T>>>
{
    pub fn log(&self) -> &[Event<BB, MM, I::StringRef, S, T>] {
        &self.log
    }
}

pub(crate) trait Log<B, M, R, S, T> {
    fn log<F: FnOnce() -> Event<B, M, R, S, T>>(&mut self, f: F);
}

impl<C, R: SpanSystem<BufId>, I: Interner, D, BB, MM, S, T> Log<BB, MM, I::StringRef, S, T>
    for CompositeSession<C, R, I, D, ()>
{
    fn log<F: FnOnce() -> Event<BB, MM, I::StringRef, S, T>>(&mut self, _: F) {}
}

#[cfg(test)]
impl<C, R: SpanSystem<BufId>, I: Interner, D, BB, MM, S, T> Log<BB, MM, I::StringRef, S, T>
    for CompositeSession<C, R, I, D, Vec<Event<BB, MM, I::StringRef, S, T>>>
{
    fn log<F: FnOnce() -> Event<BB, MM, I::StringRef, S, T>>(&mut self, f: F) {
        self.log.push(f())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Event<B, M, R, S, T> {
    AnalyzeFile {
        path: R,
        from: Option<S>,
    },
    DefineMacro {
        name_span: S,
        params: (Box<[R]>, Box<[S]>),
        body: (Box<[SemanticToken<R>]>, Box<[S]>),
    },
    DefineNameWithVisibility {
        ident: R,
        visibility: Visibility,
        entry: ResolvedName<M, B>,
    },
    DefineSymbol {
        name: B,
        span: S,
        expr: Expr<B, S>,
    },
    EmitDiag {
        diag: CompactDiag<S, T>,
    },
    EmitFragment {
        fragment: Fragment<Expr<B, S>>,
    },
    ExpandMacro {
        name: (M, S),
        args: MacroArgs<SemanticToken<R>, S>,
    },
    SetOrigin {
        addr: Expr<B, S>,
    },
    StartScope,
    StartSection {
        name: B,
        span: S,
    },
}

#[cfg(test)]
pub mod mock {
    use super::*;

    use crate::codebase::fake::FakeCodebase;
    use crate::diagnostics::mock::Merge;
    use crate::diagnostics::IgnoreDiagnostics;
    use crate::span::fake::FakeSpanSystem;

    pub type Expr<S> = crate::expr::Expr<SymbolId, S>;

    pub(in crate::assembler) type MockSession<S> = CompositeSession<
        FakeCodebase,
        FakeSpanSystem<BufId, S>,
        MockInterner,
        IgnoreDiagnostics,
        Vec<Event<SymbolId, MacroId, String, S, S>>,
    >;

    impl<S: Clone + Default + Merge> MockSession<S> {
        pub fn new() -> Self {
            let mut names = BiLevelNameTable::new();
            for (ident, keyword) in KEYWORDS {
                names
                    .global
                    .insert((*ident).into(), ResolvedName::Keyword(keyword));
            }
            CompositeSession {
                codebase: FakeCodebase::default(),
                registry: FakeSpanSystem::default(),
                interner: MockInterner,
                tokens: Vec::new(),
                macros: Vec::new(),
                names,
                builder: ObjectBuilder::new(),
                diagnostics: IgnoreDiagnostics,
                log: Vec::new(),
            }
        }

        pub fn fail(&mut self, error: CodebaseError) {
            self.codebase.fail(error)
        }
    }
}
