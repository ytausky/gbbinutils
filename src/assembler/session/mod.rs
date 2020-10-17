use self::builder::ObjectBuilder;
#[cfg(test)]
use self::macros::MacroArgs;
use self::macros::{MacroTable, VecMacroTable};
use self::resolve::*;

use super::keywords::{BuiltinMnemonic, Keyword, OperandKeyword, KEYWORDS};
use super::string_ref::StringRef;
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

mod builder;
mod lex;
mod macros;
mod reentrancy;
mod resolve;

pub(super) trait Analysis:
    SpanSource
    + NextToken
    + ReentrancyActions<<Self as SpanSource>::Span>
    + Backend<<Self as SpanSource>::Span>
    + Diagnostics<<Self as SpanSource>::Span>
    + StartScope
    + NameTable<StringRef>
    + MacroTable<<Self as SpanSource>::Span>
{
    fn mnemonic_lookup(&mut self, mnemonic: StringRef) -> Option<MnemonicEntry>;
}

pub(crate) trait NextToken: SpanSource {
    fn next_token(&mut self) -> Option<LexItem<Self::Span>>;
}

pub(super) trait ReentrancyActions<S> {
    fn analyze_file(&mut self, path: StringRef, from: Option<S>) -> Result<(), CodebaseError>;
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
    ) -> Option<NameEntry>;

    fn define_name_with_visibility(&mut self, ident: I, visibility: Visibility, entry: NameEntry);
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacroId(pub usize);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Visibility {
    Global,
    Local,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum NameEntry {
    OperandKeyword(OperandKeyword),
    Symbol(SymbolId),
}

pub(super) type Session<'a> = CompositeSession<
    FileCodebase<'a, dyn FileSystem>,
    RcContextFactory<BufId>,
    OutputForwarder<'a>,
>;

impl<C, R, D> CompositeSession<C, R, D>
where
    R: Default + SpanSystem<BufId>,
{
    pub fn new(codebase: C, diagnostics: D) -> Self {
        let mut mnemonics = HashMap::new();
        let mut names = BiLevelNameTable::new();
        for (ident, keyword) in KEYWORDS {
            let string = (*ident).into();
            match keyword {
                Keyword::BuiltinMnemonic(mnemonic) => {
                    mnemonics.insert(string, MnemonicEntry::Builtin(mnemonic));
                }
                Keyword::Operand(keyword) => {
                    names
                        .global
                        .insert(string, NameEntry::OperandKeyword(*keyword));
                }
            }
        }
        for (ident, name) in crate::eval::BUILTIN_SYMBOLS {
            names
                .global
                .insert((*ident).into(), NameEntry::Symbol(*name));
        }
        Self {
            builder: ObjectBuilder::new(),
            codebase,
            diagnostics,
            #[cfg(test)]
            log: Vec::new(),
            macros: Vec::new(),
            metadata: R::default(),
            mnemonics,
            names,
            tokens: Vec::new(),
        }
    }
}

impl<C, R, D> Analysis for CompositeSession<C, R, D>
where
    for<'a> DiagnosticsContext<'a, C, R, D>: EmitDiag<R::Span, R::Stripped>,
    C: Codebase,
    R: SpanSystem<BufId>,
    R::FileInclusionMetadataId: 'static,
    R::Span: 'static,
    R::Stripped: Clone,
{
    fn mnemonic_lookup(&mut self, mnemonic: StringRef) -> Option<MnemonicEntry> {
        if let Some(entry) = self.mnemonics.get(&mnemonic) {
            Some(entry.clone())
        } else {
            let representative = mnemonic.to_ascii_uppercase();
            if let Some(builtin @ MnemonicEntry::Builtin(_)) =
                self.mnemonics.get(representative.as_str())
            {
                let builtin = (*builtin).clone();
                Some(self.mnemonics.entry(mnemonic).or_insert(builtin).clone())
            } else {
                None
            }
        }
    }
}

pub(crate) trait TokenStream<R: SpanSource> {
    fn next_token(&mut self, registry: &mut R) -> Option<LexItem<R::Span>>;
}

pub(super) struct CompositeSession<C, R: SpanSystem<BufId>, D> {
    pub codebase: C,
    tokens: Vec<Box<dyn TokenStream<R>>>,
    macros: VecMacroTable<R::MacroDefMetadataId>,
    pub metadata: R,
    mnemonics: HashMap<StringRef, MnemonicEntry>,
    names: BiLevelNameTable<StringRef>,
    pub builder: ObjectBuilder<R::Span>,
    pub diagnostics: D,
    #[cfg(test)]
    log: Vec<Event<SymbolId, MacroId, R::Span, R::Stripped>>,
}

#[derive(Clone)]
pub(super) enum MnemonicEntry {
    Builtin(&'static BuiltinMnemonic),
    Macro(MacroId),
}

impl<C, R: SpanSystem<BufId>, D> SpanSource for CompositeSession<C, R, D> {
    type Span = R::Span;
}

impl<C, R: SpanSystem<BufId>, D> NextToken for CompositeSession<C, R, D> {
    fn next_token(&mut self) -> Option<LexItem<Self::Span>> {
        let token = self
            .tokens
            .last_mut()
            .unwrap()
            .next_token(&mut self.metadata)
            .unwrap();
        if let Ok(Token::Sigil(Sigil::Eos)) = token.0 {
            self.tokens.pop();
        }
        Some(token)
    }
}

impl<C, R: SpanSystem<BufId>, D> EmitDiag<R::Span, R::Stripped> for CompositeSession<C, R, D>
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

impl<C, R: SpanSystem<BufId>, D> CompositeSession<C, R, D> {
    fn diagnostics(&mut self) -> DiagnosticsContext<C, R, D> {
        DiagnosticsContext {
            codebase: &mut self.codebase,
            registry: &mut self.metadata,
            diagnostics: &mut self.diagnostics,
        }
    }
}

impl<C, R: SpanSystem<BufId>, D> MergeSpans<R::Span> for CompositeSession<C, R, D> {
    fn merge_spans(&mut self, left: &R::Span, right: &R::Span) -> R::Span {
        self.metadata.merge_spans(left, right)
    }
}

impl<C, R: SpanSystem<BufId>, D> StripSpan<R::Span> for CompositeSession<C, R, D> {
    type Stripped = R::Stripped;

    fn strip_span(&mut self, span: &R::Span) -> Self::Stripped {
        self.metadata.strip_span(span)
    }
}

#[cfg(test)]
impl<C, R: SpanSystem<BufId>, D> CompositeSession<C, R, D> {
    pub fn log(&self) -> &[Event<SymbolId, MacroId, R::Span, R::Stripped>] {
        &self.log
    }

    fn log_event(&mut self, event: Event<SymbolId, MacroId, R::Span, R::Stripped>) {
        self.log.push(event)
    }
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Event<B, M, S, T> {
    AnalyzeFile {
        path: StringRef,
        from: Option<S>,
    },
    DefineMacro {
        name: (StringRef, S),
        params: (Box<[StringRef]>, Box<[S]>),
        body: (Box<[SemanticToken]>, Box<[S]>),
    },
    DefineNameWithVisibility {
        ident: StringRef,
        visibility: Visibility,
        entry: NameEntry,
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
        args: MacroArgs<S>,
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
        CompositeSession<FakeCodebase, FakeSpanSystem<BufId, S>, IgnoreDiagnostics>;

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
