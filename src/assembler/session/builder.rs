use super::*;

use crate::diagnostics::{Diagnostics, DiagnosticsContext};
use crate::expr::Expr;
use crate::object::*;

pub(crate) struct ObjectBuilder<S> {
    pub content: Content<S>,
    state: Option<BuilderState<S>>,
    pub vars: VarTable,
}

enum BuilderState<S> {
    AnonSectionPrelude { addr: Option<Expr<SymbolId, S>> },
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
    fn push(&mut self, fragment: Fragment<Expr<SymbolId, S>>) {
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

    fn add_section(&mut self, symbol: Option<UserDefId>) {
        self.content
            .add_section(symbol, self.vars.alloc(), self.vars.alloc())
    }
}

impl<'a, R> Backend<R::Span> for CompositeSession<'a, R>
where
    R: SpanSystem,
    Self: Diagnostics<R::Span>,
    for<'r> DiagnosticsContext<'r, 'a, R, OutputForwarder<'a>>: Diagnostics<R::Span>,
{
    fn define_symbol(&mut self, name: SymbolId, _span: R::Span, expr: Expr<SymbolId, R::Span>) {
        #[cfg(test)]
        self.log_event(Event::DefineSymbol {
            name,
            span: _span,
            expr: expr.clone(),
        });

        let location = self.builder.vars.alloc();
        self.builder.push(Fragment::Reloc(location));
        self.builder.content.symbols.define(
            name.content().unwrap(),
            UserDef::Closure(Closure { expr, location }),
        );
    }

    fn emit_fragment(&mut self, fragment: Fragment<Expr<SymbolId, R::Span>>) {
        #[cfg(test)]
        self.log_event(Event::EmitFragment {
            fragment: fragment.clone(),
        });

        self.builder.push(fragment)
    }

    fn is_non_zero(&mut self, value: Expr<SymbolId, R::Span>) -> Option<bool> {
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

    fn set_origin(&mut self, addr: Expr<SymbolId, R::Span>) {
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

    fn start_section(&mut self, name: SymbolId, _span: R::Span) {
        #[cfg(test)]
        self.log_event(Event::StartSection { name, span: _span });

        let index = self.builder.content.sections.len();
        self.builder.state = Some(BuilderState::SectionPrelude(index));
        self.builder.add_section(Some(name.content().unwrap()))
    }
}

impl<'a, R> AllocSymbol<R::Span> for CompositeSession<'a, R>
where
    R: SpanSystem,
{
    fn alloc_symbol(&mut self, _: R::Span) -> SymbolId {
        self.builder.content.symbols.alloc().into()
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
        let mut wrapped_name = None;
        let content = build_object(|session| {
            let name = session.alloc_symbol(());
            session.start_section(name, ());
            wrapped_name = Some(name);
        });
        assert_eq!(
            content
                .symbols
                .get(wrapped_name.unwrap().content().unwrap()),
            Some(&UserDef::Section(SectionId(0)))
        )
    }

    #[test]
    fn set_origin_in_section_prelude_sets_origin() {
        let origin: Expr<_, _> = 0x0150.into();
        let content = build_object(|session| {
            let name = session.alloc_symbol(());
            session.start_section(name, ());
            session.set_origin(origin.clone())
        });
        assert_eq!(content.sections[0].constraints.addr, Some(origin))
    }

    #[test]
    fn emit_fragment_into_named_section() {
        let content = build_object(|session| {
            let name = session.alloc_symbol(());
            session.start_section(name, ());
            session.emit_fragment(Fragment::Byte(0x00))
        });
        assert_eq!(content.sections[0].fragments, [Fragment::Byte(0x00)])
    }

    fn build_object<F: FnOnce(&mut MockSession<S>), S: Clone + Default + Merge>(
        f: F,
    ) -> Content<S> {
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
}
