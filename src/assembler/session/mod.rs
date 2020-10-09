use self::builder::ObjectBuilder;
#[cfg(test)]
use self::macros::MacroArgs;
use self::macros::{MacroTable, VecMacroTable};
use self::resolve::*;

use super::keywords::{BuiltinMnemonic, OperandSymbol, KEYWORDS};
use super::semantics::Keyword;
#[cfg(test)]
use super::syntax::SemanticToken;
use super::syntax::{LexItem, Sigil, Token};

use crate::codebase::{BufId, Codebase, CodebaseError, FileCodebase, FileSystem};
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
    + NextToken
    + Interner
    + ReentrancyActions<<Self as StringSource>::StringRef, <Self as SpanSource>::Span>
    + Backend<<Self as SpanSource>::Span>
    + Diagnostics<<Self as SpanSource>::Span>
    + StartScope
    + NameTable<<Self as StringSource>::StringRef>
    + MacroTable<<Self as StringSource>::StringRef, <Self as SpanSource>::Span>
{
    fn mnemonic_lookup(&mut self, mnemonic: &Self::StringRef) -> Option<MnemonicEntry>;
}

pub(crate) trait NextToken: StringSource + SpanSource {
    fn next_token(&mut self) -> Option<LexItem<Self::StringRef, Self::Span>>;
}

pub(super) trait ReentrancyActions<R, S> {
    fn analyze_file(&mut self, path: R, from: Option<S>) -> Result<(), CodebaseError>;
}

pub(crate) trait Backend<S: Clone>: AllocSymbol<S> {
    fn define_symbol(&mut self, name: SymbolId, span: S, expr: Expr<SymbolId, S>);
    fn emit_fragment(&mut self, fragment: Fragment<Expr<SymbolId, S>>);
    fn is_non_zero(&mut self, value: Expr<SymbolId, S>) -> Option<bool>;
    fn set_origin(&mut self, origin: Expr<SymbolId, S>);
    fn start_section(&mut self, name: SymbolId, span: S);
}

pub trait AllocSymbol<S: Clone> {
    fn alloc_symbol(&mut self, span: S) -> SymbolId;
}

pub(super) trait NameTable<I> {
    fn resolve_name_with_visibility(
        &mut self,
        ident: &I,
        visibility: Visibility,
    ) -> Option<ResolvedName>;

    fn define_name_with_visibility(
        &mut self,
        ident: I,
        visibility: Visibility,
        entry: ResolvedName,
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
pub(crate) enum ResolvedName {
    Keyword(OperandSymbol),
    Symbol(SymbolId),
}

pub trait StringSource {
    type StringRef: Clone + Debug + Eq + Hash;
}

pub(super) type Session<'a> = CompositeSession<
    FileCodebase<'a, dyn FileSystem>,
    RcContextFactory<BufId>,
    HashInterner,
    OutputForwarder<'a>,
>;

impl<C, R, I, D> CompositeSession<C, R, I, D>
where
    R: Default + SpanSystem<BufId>,
    I: Default + Interner,
{
    pub fn new(codebase: C, diagnostics: D) -> Self {
        let mut interner = I::default();
        let mut mnemonics = HashMap::new();
        let mut names = BiLevelNameTable::new();
        for (ident, keyword) in KEYWORDS {
            let string = interner.intern(ident);
            match keyword {
                Keyword::BuiltinMnemonic(mnemonic) => {
                    mnemonics.insert(string, MnemonicEntry::Builtin(mnemonic));
                }
                Keyword::Operand(keyword) => {
                    names.global.insert(string, ResolvedName::Keyword(*keyword));
                }
            }
        }
        for (ident, name) in crate::eval::BUILTIN_SYMBOLS {
            let string = interner.intern(ident);
            names.global.insert(string, ResolvedName::Symbol(*name));
        }
        Self {
            builder: ObjectBuilder::new(),
            codebase,
            diagnostics,
            interner,
            #[cfg(test)]
            log: Vec::new(),
            macros: Vec::new(),
            mnemonics,
            names,
            registry: R::default(),
            tokens: Vec::new(),
        }
    }
}

impl<C, R, I, D> Analysis for CompositeSession<C, R, I, D>
where
    for<'a> DiagnosticsContext<'a, C, R, D>: EmitDiag<R::Span, R::Stripped>,
    C: Codebase,
    R: SpanSystem<BufId>,
    R::FileInclusionMetadataId: 'static,
    R::Span: 'static,
    R::Stripped: Clone,
    I: Interner,
    I::StringRef: 'static,
{
    fn mnemonic_lookup(&mut self, mnemonic: &Self::StringRef) -> Option<MnemonicEntry> {
        if let Some(entry) = self.mnemonics.get(&mnemonic) {
            Some(entry.clone())
        } else {
            let representative = self
                .interner
                .intern(&self.interner.get_string(mnemonic).to_ascii_uppercase());
            if let Some(builtin @ MnemonicEntry::Builtin(_)) = self.mnemonics.get(&representative) {
                let builtin = (*builtin).clone();
                Some(
                    self.mnemonics
                        .entry(mnemonic.clone())
                        .or_insert(builtin)
                        .clone(),
                )
            } else {
                None
            }
        }
    }
}

pub(crate) trait TokenStream<R: SpanSource, I: StringSource> {
    fn next_token(
        &mut self,
        registry: &mut R,
        interner: &mut I,
    ) -> Option<LexItem<I::StringRef, R::Span>>;
}

pub(super) struct CompositeSession<C, R: SpanSystem<BufId>, I: StringSource, D> {
    pub codebase: C,
    pub registry: R,
    interner: I,
    tokens: Vec<Box<dyn TokenStream<R, I>>>,
    macros: VecMacroTable<R::MacroDefMetadataId, I::StringRef>,
    mnemonics: HashMap<I::StringRef, MnemonicEntry>,
    names: BiLevelNameTable<I::StringRef>,
    pub builder: ObjectBuilder<R::Span>,
    pub diagnostics: D,
    #[cfg(test)]
    log: Vec<Event<SymbolId, MacroId, I::StringRef, R::Span, R::Stripped>>,
}

#[derive(Clone)]
pub(super) enum MnemonicEntry {
    Builtin(&'static BuiltinMnemonic),
    Macro(MacroId),
}

impl<C, R: SpanSystem<BufId>, I: StringSource, D> SpanSource for CompositeSession<C, R, I, D> {
    type Span = R::Span;
}

impl<C, R: SpanSystem<BufId>, I: StringSource, D> StringSource for CompositeSession<C, R, I, D> {
    type StringRef = I::StringRef;
}

impl<C, R: SpanSystem<BufId>, I: StringSource, D> NextToken for CompositeSession<C, R, I, D> {
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

impl<C, R: SpanSystem<BufId>, I: StringSource, D> EmitDiag<R::Span, R::Stripped>
    for CompositeSession<C, R, I, D>
where
    for<'a> DiagnosticsContext<'a, C, R, D>: EmitDiag<R::Span, R::Stripped>,
    R::Stripped: Clone,
{
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<R::Span, R::Stripped>>) {
        let diag = diag.into();

        #[cfg(test)]
        self.log_event(Event::EmitDiag { diag: diag.clone() });

        self.diagnostics().emit_diag(diag)
    }
}

impl<C, R: SpanSystem<BufId>, I: StringSource, D> CompositeSession<C, R, I, D> {
    fn diagnostics(&mut self) -> DiagnosticsContext<C, R, D> {
        DiagnosticsContext {
            codebase: &mut self.codebase,
            registry: &mut self.registry,
            diagnostics: &mut self.diagnostics,
        }
    }
}

impl<C, R: SpanSystem<BufId>, I: StringSource, D> MergeSpans<R::Span>
    for CompositeSession<C, R, I, D>
{
    fn merge_spans(&mut self, left: &R::Span, right: &R::Span) -> R::Span {
        self.registry.merge_spans(left, right)
    }
}

impl<C, R: SpanSystem<BufId>, I: StringSource, D> StripSpan<R::Span>
    for CompositeSession<C, R, I, D>
{
    type Stripped = R::Stripped;

    fn strip_span(&mut self, span: &R::Span) -> Self::Stripped {
        self.registry.strip_span(span)
    }
}

impl<C, R: SpanSystem<BufId>, I: Interner, D> Interner for CompositeSession<C, R, I, D> {
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

#[derive(Default)]
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

#[derive(Default)]
pub(crate) struct HashInterner {
    map: HashMap<String, StringId>,
    strings: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct StringId(usize);

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
impl<C, R: SpanSystem<BufId>, I: StringSource, D> CompositeSession<C, R, I, D> {
    pub fn log(&self) -> &[Event<SymbolId, MacroId, I::StringRef, R::Span, R::Stripped>] {
        &self.log
    }

    fn log_event(&mut self, event: Event<SymbolId, MacroId, I::StringRef, R::Span, R::Stripped>) {
        self.log.push(event)
    }
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Event<B, M, R, S, T> {
    AnalyzeFile {
        path: R,
        from: Option<S>,
    },
    DefineMacro {
        name: (R, S),
        params: (Box<[R]>, Box<[S]>),
        body: (Box<[SemanticToken<R>]>, Box<[S]>),
    },
    DefineNameWithVisibility {
        ident: R,
        visibility: Visibility,
        entry: ResolvedName,
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

    pub(in crate::assembler) type MockSession<S> =
        CompositeSession<FakeCodebase, FakeSpanSystem<BufId, S>, MockInterner, IgnoreDiagnostics>;

    impl<S: Clone + Default + Merge> Default for MockSession<S> {
        fn default() -> Self {
            Self::new(FakeCodebase::default(), IgnoreDiagnostics)
        }
    }

    impl<S: Clone + Default + Merge> MockSession<S> {
        pub fn fail(&mut self, error: CodebaseError) {
            self.codebase.fail(error)
        }
    }
}
