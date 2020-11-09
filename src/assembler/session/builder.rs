use super::*;

use crate::diagnostics::{Diagnostics, DiagnosticsContext};
use crate::expr::Expr;
use crate::object::*;

pub(crate) struct ObjectBuilder<S> {
    content: Content<StringRef, S>,
    state: Option<BuilderState<S>>,
    vars: VarTable,
}

enum BuilderState<S> {
    AnonSectionPrelude { addr: Option<Expr<Name, S>> },
    Section(usize),
    SectionPrelude(usize),
}

impl<S> ObjectBuilder<S> {
    pub fn new() -> Self {
        ObjectBuilder {
            content: Content::new(),
            state: Some(BuilderState::AnonSectionPrelude { addr: None }),
            vars: VarTable::new(),
        }
    }
}

impl<S> ObjectBuilder<S> {
    fn push(&mut self, fragment: Fragment<Expr<Name, S>>) {
        self.current_section().fragments.push(fragment)
    }

    fn current_section(&mut self) -> &mut Section<S> {
        match self.state.take().unwrap() {
            BuilderState::AnonSectionPrelude { addr } => {
                self.add_section(None);
                let index = self.content.sections.len() - 1;
                self.state = Some(BuilderState::Section(index));
                let section = &mut self.content.sections[index];
                section.constraints.addr = addr;
                section
            }
            BuilderState::SectionPrelude(index) | BuilderState::Section(index) => {
                self.state = Some(BuilderState::Section(index));
                &mut self.content.sections[index]
            }
        }
    }

    fn add_section(&mut self, symbol: Option<(SymbolId, S)>) -> SectionId {
        let section = SectionId(self.content.sections.len());
        self.content.sections.push(Section {
            constraints: Constraints { addr: None },
            addr: self.vars.alloc(),
            size: self.vars.alloc(),
            fragments: Vec::new(),
        });
        if let Some((symbol, def_ident_span)) = symbol {
            self.define_symbol(
                symbol,
                SymbolDefRecord {
                    def_ident_span,
                    meaning: SymbolMeaning::Section(section),
                },
            )
        }
        section
    }

    fn define_symbol(&mut self, SymbolId(id): SymbolId, def: SymbolDefRecord<S>) {
        match &mut self.content.symbols[id] {
            Symbol::Unknown { ident } => {
                self.content.symbols[id] = if ident.starts_with('_') {
                    Symbol::Local { def }
                } else {
                    let ident = std::mem::replace(ident, StringRef::default());
                    Symbol::Exported { ident, def }
                }
            }
            _ => todo!(),
        }
    }

    pub fn alloc_symbol(&mut self, ident: StringRef) -> Name {
        let name = Name::Symbol(SymbolId(self.content.symbols.len()));
        self.content.symbols.push(Symbol::Unknown { ident });
        name
    }

    pub fn into_content(self) -> Content<Box<str>, S> {
        Content {
            sections: self.content.sections,
            symbols: self
                .content
                .symbols
                .into_iter()
                .map(|symbol| match symbol {
                    Symbol::Exported { ident, def } => Symbol::Exported {
                        ident: ident.to_string().into_boxed_str(),
                        def,
                    },
                    Symbol::Local { def } => Symbol::Local { def },
                    Symbol::Unknown { ident } => Symbol::Unknown {
                        ident: ident.to_string().into_boxed_str(),
                    },
                })
                .collect(),
            vars: self.vars.0.len(),
        }
    }
}

impl<'a, R> Backend<R::Span> for CompositeSession<'a, R>
where
    R: SpanSystem,
    Self: Diagnostics<R::Span>,
    for<'r> DiagnosticsContext<'r, 'a, R, OutputForwarder<'a>>: Diagnostics<R::Span>,
{
    fn define_symbol(&mut self, (ident, span): (StringRef, R::Span), def: SymbolDef<R::Span>) {
        #[cfg(test)]
        self.log_event(Event::DefineSymbol {
            symbol: (ident.clone(), span.clone()),
            def: def.clone(),
        });

        match self.query_term(&ident) {
            NameEntry::Symbol(Name::Symbol(symbol)) => {
                if !ident.starts_with('_') {
                    self.names.local = HashMap::new();
                }
                match def {
                    SymbolDef::Closure(expr) => {
                        let location = self.builder.vars.alloc();
                        self.builder.push(Fragment::Reloc(location));
                        self.builder.define_symbol(
                            symbol,
                            SymbolDefRecord {
                                def_ident_span: span,
                                meaning: SymbolMeaning::Closure(Closure { expr, location }),
                            },
                        )
                    }
                    SymbolDef::Section => {
                        let section = self.builder.add_section(Some((symbol, span)));
                        self.builder.state = Some(BuilderState::SectionPrelude(section.0))
                    }
                }
            }
            NameEntry::Symbol(Name::Builtin(_)) => todo!(),
            NameEntry::OperandKeyword(_) => todo!(),
        }
    }

    fn emit_fragment(&mut self, fragment: Fragment<Expr<Name, R::Span>>) {
        #[cfg(test)]
        self.log_event(Event::EmitFragment {
            fragment: fragment.clone(),
        });

        self.builder.push(fragment)
    }

    fn is_non_zero(&mut self, value: Expr<Name, R::Span>) -> Option<bool> {
        let context = LinkageContext {
            content: &self.builder.content,
            vars: &self.builder.vars,
            location: 0.into(),
        };
        let mut diagnostics = DiagnosticsContext {
            codebase: &mut self.codebase,
            registry: &mut self.metadata,
            diagnostics: &mut self.diagnostics,
        };
        value
            .to_num(&context, &mut diagnostics)
            .exact()
            .map(|n| n != 0)
    }

    fn set_origin(&mut self, addr: Expr<Name, R::Span>) {
        #[cfg(test)]
        self.log_event(Event::SetOrigin { addr: addr.clone() });

        match self.builder.state.take().unwrap() {
            BuilderState::SectionPrelude(index) => {
                self.builder.content.sections[index].constraints.addr = Some(addr);
                self.builder.state = Some(BuilderState::SectionPrelude(index))
            }
            _ => self.builder.state = Some(BuilderState::AnonSectionPrelude { addr: Some(addr) }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::assembler::session::mock::MockSession;
    use crate::expr::{Atom, ExprOp};
    use crate::object::SectionId;
    use crate::span::WithSpan;

    #[test]
    fn new_object_has_no_sections() {
        let content = build_object::<_, ()>(|_| ());
        assert_eq!(content.sections.len(), 0)
    }

    #[test]
    fn no_origin_by_default() {
        let content = build_object::<_, ()>(|session| session.emit_fragment(Fragment::Byte(0x00)));
        assert_eq!(content.sections[0].constraints.addr, None)
    }

    #[test]
    fn constrain_origin_determines_origin_of_new_section() {
        let origin: Expr<_, _> = 0x3000.into();
        let content = build_object(|session| {
            session.set_origin(origin.clone());
            session.emit_fragment(Fragment::Byte(0x00))
        });
        assert_eq!(content.sections[0].constraints.addr, Some(origin))
    }

    #[test]
    fn start_section_adds_named_section() {
        let ident = StringRef::from("my_section");
        let content =
            build_object(|session| session.define_symbol((ident.clone(), ()), SymbolDef::Section));
        assert_eq!(
            content.symbols[0],
            Symbol::Exported {
                ident,
                def: SymbolDefRecord {
                    def_ident_span: (),
                    meaning: SymbolMeaning::Section(SectionId(0))
                }
            }
        )
    }

    #[test]
    fn set_origin_in_section_prelude_sets_origin() {
        let origin: Expr<_, _> = 0x0150.into();
        let content = build_object(|session| {
            session.define_symbol(("my_section".into(), ()), SymbolDef::Section);
            session.set_origin(origin.clone())
        });
        assert_eq!(content.sections[0].constraints.addr, Some(origin))
    }

    #[test]
    fn emit_fragment_into_named_section() {
        let content = build_object(|session| {
            session.define_symbol(("my_section".into(), ()), SymbolDef::Section);
            session.emit_fragment(Fragment::Byte(0x00))
        });
        assert_eq!(content.sections[0].fragments, [Fragment::Byte(0x00)])
    }

    fn build_object<F: FnOnce(&mut MockSession<S>), S: Clone + Default + Merge>(
        f: F,
    ) -> Content<StringRef, S> {
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        f(&mut session);
        session.builder.content
    }

    #[test]
    fn reserve_bytes_in_section() {
        let bytes = 3;
        let content =
            build_object(|builder| builder.emit_fragment(Fragment::Reserved(bytes.into())));
        assert_eq!(
            content.sections[0].fragments,
            [Fragment::Reserved(bytes.into())]
        )
    }

    #[test]
    fn eval_zero() {
        build_object(|object_builder| {
            assert_eq!(
                object_builder.is_non_zero(Expr(vec![ExprOp::Atom(Atom::Const(0)).with_span(())])),
                Some(false)
            )
        });
    }

    #[test]
    fn eval_42() {
        build_object(|object_builder| {
            assert_eq!(
                object_builder.is_non_zero(Expr(vec![ExprOp::Atom(Atom::Const(42)).with_span(())])),
                Some(true)
            )
        });
    }

    #[test]
    fn local_symbol_not_accessible_after_section_definition() {
        let name = "_local".into();
        let mut fixture = TestFixture::<()>::new();
        let mut session = fixture.session();
        let entry1 = session.query_term(&name);
        session.define_symbol(("my_section".into(), ()), SymbolDef::Section);
        let entry2 = session.query_term(&name);
        assert_ne!(entry1, entry2)
    }

    #[test]
    fn local_symbol_not_accessible_after_closure_definition() {
        let name = "_local".into();
        let mut fixture = TestFixture::<()>::new();
        let mut session = fixture.session();
        let entry1 = session.query_term(&name);
        session.define_symbol(
            ("label".into(), ()),
            SymbolDef::Closure(Expr::from_atom(Atom::Location, ())),
        );
        let entry2 = session.query_term(&name);
        assert_ne!(entry1, entry2)
    }

    #[test]
    fn symbol_starting_with_underscore_is_local() {
        let mut fixture = TestFixture::<()>::new();
        let mut session = fixture.session();
        session.define_symbol(
            ("_loop".into(), ()),
            SymbolDef::Closure(Expr::from_atom(Atom::Location, ())),
        );
        assert!(matches!(session.builder.content.symbols[0], Symbol::Local { .. }))
    }
}
