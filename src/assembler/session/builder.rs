use super::*;

use crate::diagnostics::{Diagnostics, DiagnosticsContext};
use crate::expr::Expr;
use crate::object::*;

pub(crate) struct ObjectBuilder<S> {
    pub data: Data<S>,
    state: Option<BuilderState<S>>,
}

enum BuilderState<S> {
    AnonSectionPrelude { addr: Option<Expr<SymbolId, S>> },
    Section(usize),
    SectionPrelude(usize),
}

impl<S> ObjectBuilder<S> {
    pub fn new() -> Self {
        ObjectBuilder {
            data: Data::new(),
            state: Some(BuilderState::AnonSectionPrelude { addr: None }),
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
                let index = self.data.content.sections.len() - 1;
                self.state = Some(BuilderState::Section(index));
                let section = &mut self.data.content.sections[index];
                section.constraints.addr = addr;
                section
            }
            BuilderState::SectionPrelude(index) | BuilderState::Section(index) => {
                self.state = Some(BuilderState::Section(index));
                &mut self.data.content.sections[index]
            }
        }
    }

    fn add_section(&mut self, symbol: Option<UserDefId>) {
        self.data
            .content
            .add_section(symbol, self.data.vars.alloc(), self.data.vars.alloc())
    }
}

impl<C, R, D> Backend<R::Span> for CompositeSession<C, R, D>
where
    R: SpanSystem<BufId>,
    Self: Diagnostics<R::Span>,
    for<'a> DiagnosticsContext<'a, C, R, D>: Diagnostics<R::Span>,
{
    fn define_symbol(&mut self, name: SymbolId, _span: R::Span, expr: Expr<SymbolId, R::Span>) {
        #[cfg(test)]
        self.log_event(Event::DefineSymbol {
            name,
            span: _span,
            expr: expr.clone(),
        });

        let location = self.builder.data.vars.alloc();
        self.builder.push(Fragment::Reloc(location));
        self.builder.data.content.symbols.define(
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
            content: &self.builder.data.content,
            vars: &self.builder.data.vars,
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
                self.builder.data.content.sections[index].constraints.addr = Some(addr);
                self.builder.state = Some(BuilderState::SectionPrelude(index))
            }
            _ => self.builder.state = Some(BuilderState::AnonSectionPrelude { addr: Some(addr) }),
        }
    }

    fn start_section(&mut self, name: SymbolId, _span: R::Span) {
        #[cfg(test)]
        self.log_event(Event::StartSection { name, span: _span });

        let index = self.builder.data.content.sections.len();
        self.builder.state = Some(BuilderState::SectionPrelude(index));
        self.builder.add_section(Some(name.content().unwrap()))
    }
}

impl<C, R, D> AllocSymbol<R::Span> for CompositeSession<C, R, D>
where
    R: SpanSystem<BufId>,
{
    fn alloc_symbol(&mut self, _: R::Span) -> SymbolId {
        self.builder.data.content.symbols.alloc().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::assembler::session::mock::MockSession;
    use crate::diagnostics::*;
    use crate::expr::{Atom, ExprOp};
    use crate::object::SectionId;
    use crate::span::WithSpan;

    #[test]
    fn new_object_has_no_sections() {
        let data = build_object::<_, ()>(|_| ());
        assert_eq!(data.content.sections.len(), 0)
    }

    #[test]
    fn no_origin_by_default() {
        let data = build_object::<_, ()>(|session| session.emit_fragment(Fragment::Byte(0x00)));
        assert_eq!(data.content.sections[0].constraints.addr, None)
    }

    #[test]
    fn constrain_origin_determines_origin_of_new_section() {
        let origin: Expr<_, _> = 0x3000.into();
        let data = build_object(|session| {
            session.set_origin(origin.clone());
            session.emit_fragment(Fragment::Byte(0x00))
        });
        assert_eq!(data.content.sections[0].constraints.addr, Some(origin))
    }

    #[test]
    fn start_section_adds_named_section() {
        let mut wrapped_name = None;
        let data = build_object(|session| {
            let name = session.alloc_symbol(());
            session.start_section(name, ());
            wrapped_name = Some(name);
        });
        assert_eq!(
            data.content
                .symbols
                .get(wrapped_name.unwrap().content().unwrap()),
            Some(&UserDef::Section(SectionId(0)))
        )
    }

    #[test]
    fn set_origin_in_section_prelude_sets_origin() {
        let origin: Expr<_, _> = 0x0150.into();
        let data = build_object(|session| {
            let name = session.alloc_symbol(());
            session.start_section(name, ());
            session.set_origin(origin.clone())
        });
        assert_eq!(data.content.sections[0].constraints.addr, Some(origin))
    }

    #[test]
    fn emit_fragment_into_named_section() {
        let data = build_object(|session| {
            let name = session.alloc_symbol(());
            session.start_section(name, ());
            session.emit_fragment(Fragment::Byte(0x00))
        });
        assert_eq!(data.content.sections[0].fragments, [Fragment::Byte(0x00)])
    }

    fn build_object<F: FnOnce(&mut MockSession<S>), S: Clone + Default + Merge>(f: F) -> Data<S> {
        let mut session = MockSession::default();
        f(&mut session);
        session.builder.data
    }

    #[test]
    fn reserve_bytes_in_section() {
        let bytes = 3;
        let program =
            build_object(|builder| builder.emit_fragment(Fragment::Reserved(bytes.into())));
        assert_eq!(
            program.content.sections[0].fragments,
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
