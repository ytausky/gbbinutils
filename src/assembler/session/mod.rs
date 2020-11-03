use self::builder::ObjectBuilder;
#[cfg(test)]
use self::macros::MacroArgs;
use self::macros::{MacroTable, VecMacroTable};
#[cfg(test)]
use self::mock::MockSession;
use self::resolve::*;

use super::keywords::{BuiltinMnemonic, Keyword, OperandKeyword, KEYWORDS};
use super::string_ref::StringRef;
#[cfg(test)]
use super::syntax::SemanticToken;
use super::syntax::{LexItem, Sigil, Token};

#[cfg(test)]
use crate::codebase::fake::MockFileSystem;
use crate::codebase::{Codebase, CodebaseError, FileSystem};
use crate::diagnostics::*;
use crate::expr::Expr;
use crate::object::{Fragment, Metadata, Name, ObjectData, SpanData};
use crate::span::*;

use std::collections::HashMap;
use std::fmt::Debug;
#[cfg(test)]
use std::marker::PhantomData;

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
    + IdentTable
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

pub(super) trait Backend<S: Clone> {
    fn define_symbol(&mut self, symbol: (StringRef, S), def: SymbolDef<S>);
    fn emit_fragment(&mut self, fragment: Fragment<Expr<Name, S>>);
    fn is_non_zero(&mut self, value: Expr<Name, S>) -> Option<bool>;
    fn set_origin(&mut self, origin: Expr<Name, S>);
}

pub(super) trait IdentTable {
    fn query_term(&mut self, ident: &StringRef) -> NameEntry;
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum SymbolDef<S> {
    Closure(Expr<Name, S>),
    Section,
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
    Symbol(Name),
}

pub(super) type Session<'a> = CompositeSession<'a, SpanData>;

impl<'a, R> CompositeSession<'a, R>
where
    R: Default + SpanSystem,
{
    pub fn new(fs: &'a mut dyn FileSystem, diagnostics: &'a mut dyn FnMut(Diagnostic)) -> Self {
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
            codebase: Codebase::new(fs),
            diagnostics: OutputForwarder {
                output: diagnostics,
            },
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

impl<'a> CompositeSession<'a, SpanData> {
    pub fn try_into_object_data(self) -> ObjectData<Metadata> {
        ObjectData {
            content: self.builder.content,
            metadata: Metadata {
                source_files: self.codebase.export_source_file_table(),
                span_data: self.metadata,
            },
        }
    }
}

impl<'a, R> Analysis for CompositeSession<'a, R>
where
    for<'r> DiagnosticsContext<'r, 'a, R, OutputForwarder<'a>>: EmitDiag<R::Span, R::Stripped>,
    R: SpanSystem,
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

pub(super) struct CompositeSession<'a, R: SpanSystem> {
    codebase: Codebase<'a>,
    tokens: Vec<Box<dyn TokenStream<R>>>,
    macros: VecMacroTable,
    metadata: R,
    mnemonics: HashMap<StringRef, MnemonicEntry>,
    names: BiLevelNameTable<StringRef>,
    builder: ObjectBuilder<R::Span>,
    diagnostics: OutputForwarder<'a>,
    #[cfg(test)]
    log: Vec<Event<Name, MacroId, R::Span, R::Stripped>>,
}

#[derive(Clone)]
pub(super) enum MnemonicEntry {
    Builtin(&'static BuiltinMnemonic),
    Macro(MacroId),
}

impl<'a, R: SpanSystem> SpanSource for CompositeSession<'a, R> {
    type Span = R::Span;
}

impl<'a, R: SpanSystem> NextToken for CompositeSession<'a, R> {
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

impl<'a, R: SpanSystem> EmitDiag<R::Span, R::Stripped> for CompositeSession<'a, R>
where
    for<'r> DiagnosticsContext<'r, 'a, R, OutputForwarder<'a>>: EmitDiag<R::Span, R::Stripped>,
    R::Stripped: Clone,
{
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<R::Span, R::Stripped>>) {
        let diag = diag.into();

        #[cfg(test)]
        self.log_event(Event::EmitDiag { diag: diag.clone() });

        self.diagnostics().emit_diag(diag)
    }
}

impl<'a, R: SpanSystem> CompositeSession<'a, R> {
    fn diagnostics<'r>(&'r mut self) -> DiagnosticsContext<'r, 'a, R, OutputForwarder<'a>> {
        DiagnosticsContext {
            codebase: &mut self.codebase,
            registry: &mut self.metadata,
            diagnostics: &mut self.diagnostics,
        }
    }
}

impl<'a, R: SpanSystem> MergeSpans<R::Span> for CompositeSession<'a, R> {
    fn merge_spans(&mut self, left: &R::Span, right: &R::Span) -> R::Span {
        self.metadata.merge_spans(left, right)
    }
}

impl<'a, R: SpanSystem> StripSpan<R::Span> for CompositeSession<'a, R> {
    type Stripped = R::Stripped;

    fn strip_span(&mut self, span: &R::Span) -> Self::Stripped {
        self.metadata.strip_span(span)
    }
}

#[cfg(test)]
impl<'a, R: SpanSystem> CompositeSession<'a, R> {
    pub fn log(&self) -> &[Event<Name, MacroId, R::Span, R::Stripped>] {
        &self.log
    }

    fn log_event(&mut self, event: Event<Name, MacroId, R::Span, R::Stripped>) {
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
    DefineSymbol {
        symbol: (StringRef, S),
        def: SymbolDef<S>,
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
}

#[cfg(test)]
pub(super) struct TestFixture<S> {
    pub fs: MockFileSystem,
    drop: fn(Diagnostic),
    _phantom_data: PhantomData<S>,
}

#[cfg(test)]
impl<S: Clone + Default + Merge> TestFixture<S> {
    pub fn new() -> Self {
        Self {
            fs: MockFileSystem::new(),
            drop,
            _phantom_data: PhantomData,
        }
    }

    pub fn session(&mut self) -> MockSession<S> {
        MockSession::new(&mut self.fs, &mut self.drop)
    }
}

#[cfg(test)]
pub mod mock {
    use super::*;

    use crate::span::fake::FakeSpanSystem;

    pub type Expr<S> = crate::expr::Expr<Name, S>;

    pub(in crate::assembler) type MockSession<'a, S> = CompositeSession<'a, FakeSpanSystem<S>>;
}
