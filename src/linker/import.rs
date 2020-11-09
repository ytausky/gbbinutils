use super::Session;

use crate::expr::{Atom, ExprOp};
use crate::object::*;
#[cfg(test)]
use crate::span::fake::FakeSpanSystem;
use crate::span::SpanSource;

use std::hash::Hash;
#[cfg(test)]
use std::marker::PhantomData;

impl<'a, D, M: Default + SpanSource, I: Clone + Eq + Hash> Session<'a, D, M, I> {
    pub fn import_object<N>(&mut self, object: ObjectData<N, I>)
    where
        N: SpanSource<Span = M::Span>,
        Self: ImportMetadata<N>,
        <Self as ImportMetadata<N>>::SpanPatcher: PatchSpan<M::Span>,
    {
        let patcher = self.import_metadata(object.metadata);
        self.import_content(object.content, patcher);
    }

    fn import_content<P: PatchSpan<M::Span>>(&mut self, content: Content<I, M::Span>, patcher: P) {
        // Import symbol identifiers and create symbol mapping
        let mut symbol_mapping = Vec::with_capacity(content.symbols.len());
        for symbol in content.symbols {
            let global_id = self.content.symbols.len();
            match &symbol {
                Symbol::Exported { ident, .. } => match self.idents.get(ident) {
                    Some(SymbolId(id)) => {
                        self.content.symbols[*id] = symbol;
                        symbol_mapping.push(*id)
                    }
                    None => {
                        self.idents.insert(ident.clone(), SymbolId(global_id));
                        symbol_mapping.push(global_id);
                        self.content.symbols.push(symbol)
                    }
                },
                Symbol::Local { .. } => {
                    symbol_mapping.push(global_id);
                    self.content.symbols.push(symbol)
                }
                Symbol::Unknown { ident } => match self.idents.get(&ident) {
                    Some(id) => symbol_mapping.push(id.0),
                    None => {
                        self.idents.insert(ident.clone(), SymbolId(global_id));
                        symbol_mapping.push(global_id);
                        self.content.symbols.push(symbol)
                    }
                },
            }
        }

        let patcher = self.content_patcher(patcher, symbol_mapping.into());

        // Import sections
        self.content.sections.reserve(content.sections.len());
        for mut section in content.sections {
            patcher.patch_section(&mut section);
            self.content.sections.push(section)
        }

        // Import symbols
        for id in patcher.symbol_mapping.iter() {
            patcher.patch_symbol(&mut self.content.symbols[*id])
        }

        // Import variables
        self.content.vars += content.vars;
    }

    fn content_patcher<P>(
        &self,
        span_patcher: P,
        symbol_mapping: Box<[usize]>,
    ) -> ContentPatcher<P> {
        ContentPatcher {
            base_section: self.content.sections.len(),
            base_var: self.content.vars,
            span_patcher,
            symbol_mapping,
        }
    }
}

pub(super) trait ImportMetadata<M: SpanSource> {
    type SpanPatcher: PatchSpan<M::Span>;
    fn import_metadata(&mut self, metadata: M) -> Self::SpanPatcher;
}

pub(super) trait PatchSpan<S> {
    fn patch_span(&self, span: &mut S);
}

impl<'a, D, I> ImportMetadata<Metadata> for Session<'a, D, SpanData, I> {
    type SpanPatcher = SpanPatcher;

    fn import_metadata(&mut self, metadata: Metadata) -> Self::SpanPatcher {
        let base_source_file = self.source_file_count;
        let base_source_file_inclusion = self.metadata.source_file_inclusions.len();
        let base_macro_def = self.metadata.macro_defs.len();
        let base_macro_expansion = self.metadata.macro_expansions.len();

        // Import source files
        for path in Vec::from(metadata.source_files) {
            self.codebase.open(&path).unwrap();
        }

        let patcher = SpanPatcher {
            base_file_inclusion: base_source_file_inclusion,
            base_macro_expansion,
        };

        // Import source file inclusions
        for mut source_file_inclusion in metadata.span_data.source_file_inclusions {
            source_file_inclusion.file.0 += base_source_file;
            source_file_inclusion
                .from
                .as_mut()
                .map(|span| patcher.patch_span(span));
            self.metadata
                .source_file_inclusions
                .push(source_file_inclusion)
        }

        // Import macro definitions
        for mut macro_def in metadata.span_data.macro_defs {
            patcher.patch_span(&mut macro_def.name_span);
            for param_span in macro_def.param_spans.iter_mut() {
                patcher.patch_span(param_span)
            }
            for body_span in macro_def.body_spans.iter_mut() {
                patcher.patch_span(body_span)
            }
            self.metadata.macro_defs.push(macro_def)
        }

        // Import macro expansions
        for mut macro_expansion in metadata.span_data.macro_expansions {
            macro_expansion.def.0 += base_macro_def;
            patcher.patch_span(&mut macro_expansion.name_span);
            for arg in macro_expansion.arg_spans.iter_mut() {
                for span in arg.iter_mut() {
                    patcher.patch_span(span)
                }
            }
            self.metadata.macro_expansions.push(macro_expansion)
        }

        patcher
    }
}

pub(super) struct SpanPatcher {
    base_file_inclusion: usize,
    base_macro_expansion: usize,
}

impl PatchSpan<Span> for SpanPatcher {
    fn patch_span(&self, span: &mut Span) {
        match span {
            Span::SourceFile {
                inclusion_metadata, ..
            } => inclusion_metadata.0 += self.base_file_inclusion,
            Span::MacroExpansion { metadata, .. } => metadata.0 += self.base_macro_expansion,
        }
    }
}

struct ContentPatcher<P> {
    base_section: usize,
    base_var: usize,
    span_patcher: P,
    symbol_mapping: Box<[usize]>,
}

impl<P> ContentPatcher<P> {
    fn patch_section<S>(&self, section: &mut Section<S>)
    where
        P: PatchSpan<S>,
    {
        section
            .constraints
            .addr
            .as_mut()
            .map(|expr| self.patch_expr(expr));
        self.patch_var(&mut section.addr);
        self.patch_var(&mut section.size);
        for fragment in &mut section.fragments {
            match fragment {
                Fragment::Immediate(expr, _)
                | Fragment::Embedded(_, expr)
                | Fragment::LdInlineAddr(_, expr)
                | Fragment::Reserved(expr) => self.patch_expr(expr),
                Fragment::Reloc(var) => self.patch_var(var),
                Fragment::Byte(_) => (),
            }
        }
    }

    fn patch_symbol<I, S>(&self, symbol: &mut Symbol<I, S>)
    where
        P: PatchSpan<S>,
    {
        match symbol {
            Symbol::Exported { def, .. } => self.patch_symbol_def(def),
            Symbol::Local { def } => self.patch_symbol_def(def),
            Symbol::Unknown { .. } => (),
        }
    }

    fn patch_symbol_def<S>(&self, symbol_def: &mut SymbolDefRecord<S>)
    where
        P: PatchSpan<S>,
    {
        self.span_patcher.patch_span(&mut symbol_def.def_ident_span);
        match &mut symbol_def.meaning {
            SymbolMeaning::Closure(closure) => {
                self.patch_expr(&mut closure.expr);
                self.patch_var(&mut closure.location)
            }
            SymbolMeaning::Section(section) => section.0 += self.base_section,
        }
    }

    fn patch_expr<S>(&self, expr: &mut Expr<S>)
    where
        P: PatchSpan<S>,
    {
        for node in &mut expr.0 {
            match &mut node.item {
                ExprOp::Atom(Atom::Name(Name::Symbol(SymbolId(id)))) => {
                    *id = self.symbol_mapping[*id]
                }
                _ => (),
            }
            self.span_patcher.patch_span(&mut node.span)
        }
    }

    fn patch_var(&self, var: &mut VarId) {
        var.0 += self.base_var
    }
}

#[cfg(test)]
pub struct FakeMetadata<S>(PhantomData<S>);

#[cfg(test)]
impl<S> FakeMetadata<S> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

#[cfg(test)]
impl<S: Clone> SpanSource for FakeMetadata<S> {
    type Span = S;
}

#[cfg(test)]
impl<'a, D, S: Clone, I> ImportMetadata<FakeMetadata<S>> for Session<'a, D, FakeSpanSystem<S>, I> {
    type SpanPatcher = FakeSpanPatcher;

    fn import_metadata(&mut self, _: FakeMetadata<S>) -> Self::SpanPatcher {
        FakeSpanPatcher
    }
}

#[cfg(test)]
pub(super) struct FakeSpanPatcher;

#[cfg(test)]
impl<S> PatchSpan<S> for FakeSpanPatcher {
    fn patch_span(&self, _: &mut S) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::codebase::fake::MockFileSystem;
    use crate::diagnostics::IgnoreDiagnostics;
    use crate::expr::Atom;

    #[test]
    fn import_symbol_required_by_object1_is_exported_from_object2() {
        let ident = "my_symbol";

        // DW   my_symbol
        let object1 = ObjectData {
            content: Content {
                sections: vec![Section {
                    constraints: Constraints { addr: None },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![Fragment::Immediate(
                        Expr::from_atom(Atom::Name(Name::Symbol(SymbolId(0))), ()),
                        Width::Word,
                    )],
                }],
                symbols: vec![Symbol::Unknown { ident }],
                vars: 2,
            },
            metadata: FakeMetadata::new(),
        };

        // my_symbol
        let object2 = ObjectData {
            content: Content {
                sections: vec![Section {
                    constraints: Constraints { addr: None },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![Fragment::Reloc(VarId(2))],
                }],
                symbols: vec![Symbol::Exported {
                    ident,
                    def: SymbolDefRecord {
                        def_ident_span: (),
                        meaning: SymbolMeaning::Closure(Closure {
                            expr: Expr::from_atom(Atom::Location, ()),
                            location: VarId(2),
                        }),
                    },
                }],
                vars: 3,
            },
            metadata: FakeMetadata::new(),
        };

        let fs = &mut MockFileSystem::new();
        let mut session = Session::<_, FakeSpanSystem<_>, _>::new(fs, IgnoreDiagnostics);
        session.import_object(object1);
        session.import_object(object2);
        assert_eq!(
            session.content.symbols,
            vec![Symbol::Exported {
                ident,
                def: SymbolDefRecord {
                    def_ident_span: (),
                    meaning: SymbolMeaning::Closure(Closure {
                        expr: Expr::from_atom(Atom::Location, ()),
                        location: VarId(4),
                    }),
                }
            }]
        )
    }
}
