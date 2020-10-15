use super::*;

use crate::diagnostics::{Diagnostics, DiagnosticsContext};
use crate::expr::Expr;
use crate::object::*;

pub(crate) struct ObjectBuilder<M: SpanSource> {
    pub object: Object<M>,
    state: Option<BuilderState<M::Span>>,
}

enum BuilderState<S> {
    AnonSectionPrelude { addr: Option<Expr<SymbolId, S>> },
    Section(usize),
    SectionPrelude(usize),
}

impl<M: Default + SpanSource> ObjectBuilder<M> {
    pub fn new() -> Self {
        ObjectBuilder {
            object: Object::new(),
            state: Some(BuilderState::AnonSectionPrelude { addr: None }),
        }
    }
}

impl<M: SpanSource> ObjectBuilder<M> {
    fn push(&mut self, fragment: Fragment<Expr<SymbolId, M::Span>>) {
        self.current_section().fragments.push(fragment)
    }

    fn current_section(&mut self) -> &mut Section<M::Span> {
        match self.state.take().unwrap() {
            BuilderState::AnonSectionPrelude { addr } => {
                self.add_section(None);
                let index = self.object.content.sections.len() - 1;
                self.state = Some(BuilderState::Section(index));
                let section = &mut self.object.content.sections[index];
                section.constraints.addr = addr;
                section
            }
            BuilderState::SectionPrelude(index) | BuilderState::Section(index) => {
                self.state = Some(BuilderState::Section(index));
                &mut self.object.content.sections[index]
            }
        }
    }

    fn add_section(&mut self, symbol: Option<UserDefId>) {
        self.object
            .content
            .add_section(symbol, self.object.vars.alloc(), self.object.vars.alloc())
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

        let location = self.builder.object.vars.alloc();
        self.builder.push(Fragment::Reloc(location));
        self.builder.object.content.symbols.define(
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
            content: &self.builder.object.content,
            vars: &self.builder.object.vars,
            location: 0.into(),
        };
        let mut diagnostics = DiagnosticsContext {
            codebase: &mut self.codebase,
            registry: &mut self.builder.object.metadata,
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
                self.builder.object.content.sections[index].constraints.addr = Some(addr);
                self.builder.state = Some(BuilderState::SectionPrelude(index))
            }
            _ => self.builder.state = Some(BuilderState::AnonSectionPrelude { addr: Some(addr) }),
        }
    }

    fn start_section(&mut self, name: SymbolId, _span: R::Span) {
        #[cfg(test)]
        self.log_event(Event::StartSection { name, span: _span });

        let index = self.builder.object.content.sections.len();
        self.builder.state = Some(BuilderState::SectionPrelude(index));
        self.builder.add_section(Some(name.content().unwrap()))
    }
}

impl<C, R, D> AllocSymbol<R::Span> for CompositeSession<C, R, D>
where
    R: SpanSystem<BufId>,
{
    fn alloc_symbol(&mut self, _: R::Span) -> SymbolId {
        self.builder.object.content.symbols.alloc().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::assembler::session::mock::MockSession;
    use crate::diagnostics::*;
    use crate::expr::{Atom, BinOp, ExprOp};
    use crate::link::Program;
    use crate::object::SectionId;
    use crate::span::fake::FakeSpanSystem;
    use crate::span::WithSpan;

    use std::borrow::Borrow;

    #[test]
    fn new_object_has_no_sections() {
        let object = build_object::<_, ()>(|_| ());
        assert_eq!(object.content.sections.len(), 0)
    }

    #[test]
    fn no_origin_by_default() {
        let object = build_object::<_, ()>(|session| session.emit_fragment(Fragment::Byte(0x00)));
        assert_eq!(object.content.sections[0].constraints.addr, None)
    }

    #[test]
    fn constrain_origin_determines_origin_of_new_section() {
        let origin: Expr<_, _> = 0x3000.into();
        let object = build_object(|session| {
            session.set_origin(origin.clone());
            session.emit_fragment(Fragment::Byte(0x00))
        });
        assert_eq!(object.content.sections[0].constraints.addr, Some(origin))
    }

    #[test]
    fn start_section_adds_named_section() {
        let mut wrapped_name = None;
        let object = build_object(|session| {
            let name = session.alloc_symbol(());
            session.start_section(name, ());
            wrapped_name = Some(name);
        });
        assert_eq!(
            object
                .content
                .symbols
                .get(wrapped_name.unwrap().content().unwrap()),
            Some(&UserDef::Section(SectionId(0)))
        )
    }

    #[test]
    fn set_origin_in_section_prelude_sets_origin() {
        let origin: Expr<_, _> = 0x0150.into();
        let object = build_object(|session| {
            let name = session.alloc_symbol(());
            session.start_section(name, ());
            session.set_origin(origin.clone())
        });
        assert_eq!(object.content.sections[0].constraints.addr, Some(origin))
    }

    #[test]
    fn emit_fragment_into_named_section() {
        let object = build_object(|session| {
            let name = session.alloc_symbol(());
            session.start_section(name, ());
            session.emit_fragment(Fragment::Byte(0x00))
        });
        assert_eq!(object.content.sections[0].fragments, [Fragment::Byte(0x00)])
    }

    fn build_object<F: FnOnce(&mut MockSession<S>), S: Clone + Default + Merge>(
        f: F,
    ) -> Object<FakeSpanSystem<BufId, S>> {
        let mut session = MockSession::default();
        f(&mut session);
        session.builder.object
    }

    fn emit_items_and_compare<I, B>(items: I, bytes: B)
    where
        I: Borrow<[Fragment<Expr<SymbolId, ()>>]>,
        B: Borrow<[u8]>,
    {
        let (object, _) = with_object_builder(|builder| {
            for item in items.borrow() {
                builder.emit_fragment(item.clone())
            }
        });
        assert_eq!(object.sections.last().unwrap().data, bytes.borrow())
    }

    #[test]
    fn emit_literal_byte_item() {
        emit_items_and_compare([byte_literal(0xff)], [0xff])
    }

    #[test]
    fn emit_two_literal_byte_item() {
        emit_items_and_compare([byte_literal(0x12), byte_literal(0x34)], [0x12, 0x34])
    }

    fn byte_literal(value: i32) -> Fragment<Expr<SymbolId, ()>> {
        Fragment::Immediate(value.into(), Width::Byte)
    }

    #[test]
    fn emit_diagnostic_when_byte_item_out_of_range() {
        test_diagnostic_for_out_of_range_byte(i32::from(i8::min_value()) - 1);
        test_diagnostic_for_out_of_range_byte(i32::from(u8::max_value()) + 1)
    }

    fn test_diagnostic_for_out_of_range_byte(value: i32) {
        let (_, diagnostics) =
            with_object_builder(|builder| builder.emit_fragment(byte_literal(value)));
        assert_eq!(
            *diagnostics,
            [Message::ValueOutOfRange {
                value,
                width: Width::Byte,
            }
            .at(())
            .into()]
        );
    }

    #[test]
    fn diagnose_unresolved_symbol() {
        let name = "ident";
        let (_, diagnostics) = with_object_builder::<MockSpan<String>, _>(|builder| {
            let symbol_id = builder.alloc_symbol(name.to_string().into());
            builder.emit_fragment(word_item(Expr(vec![
                ExprOp::Atom(Atom::Name(symbol_id)).with_span(name.to_string().into())
            ])))
        });
        assert_eq!(*diagnostics, [unresolved(name)]);
    }

    #[test]
    fn diagnose_two_unresolved_symbols_in_one_expr() {
        let name1 = "ident1";
        let name2 = "ident2";
        let (_, diagnostics) = with_object_builder::<MockSpan<String>, _>(|builder| {
            let id1 = builder.alloc_symbol(name1.to_string().into());
            let id2 = builder.alloc_symbol(name2.to_string().into());
            builder.emit_fragment(word_item(Expr(vec![
                ExprOp::Atom(Atom::Name(id1)).with_span(name1.to_string().into()),
                ExprOp::Atom(Atom::Name(id2)).with_span(name2.to_string().into()),
                ExprOp::Binary(BinOp::Minus).with_span("diff".to_string().into()),
            ])))
        });
        assert_eq!(*diagnostics, [unresolved(name1), unresolved(name2)]);
    }

    #[test]
    fn emit_defined_symbol() {
        let (object, diagnostics) = with_object_builder(|builder| {
            let symbol_id = builder.alloc_symbol(());
            builder.define_symbol(
                symbol_id,
                (),
                Expr(vec![ExprOp::Atom(Atom::Location).with_span(())]),
            );
            builder.emit_fragment(word_item(Expr(vec![
                ExprOp::Atom(Atom::Name(symbol_id)).with_span(())
            ])));
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x00, 0x00])
    }

    #[test]
    fn emit_symbol_defined_after_use() {
        let (object, diagnostics) = with_object_builder(|builder| {
            let symbol_id = builder.alloc_symbol(());
            builder.emit_fragment(word_item(Expr(vec![
                ExprOp::Atom(Atom::Name(symbol_id)).with_span(())
            ])));
            builder.define_symbol(
                symbol_id,
                (),
                Expr(vec![ExprOp::Atom(Atom::Location).with_span(())]),
            );
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x02, 0x00])
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

    fn with_object_builder<S, F>(f: F) -> (Program, Box<[CompactDiag<S, S>]>)
    where
        S: Clone + Debug + Default + Merge + 'static,
        F: FnOnce(&mut MockSession<S>),
    {
        let diagnostics = TestDiagnosticsListener::new();
        let log = diagnostics.diagnostics.clone();
        let object = Program::link(build_object(f), (), diagnostics);
        let diagnostics = log.into_inner().into_boxed_slice();
        (object, diagnostics)
    }

    fn word_item<S: Clone>(value: Expr<SymbolId, S>) -> Fragment<Expr<SymbolId, S>> {
        Fragment::Immediate(value, Width::Word)
    }

    fn unresolved(symbol: impl Into<String>) -> CompactDiag<MockSpan<String>, MockSpan<String>> {
        let symbol = symbol.into();
        Message::UnresolvedSymbol {
            symbol: symbol.clone().into(),
        }
        .at(symbol.into())
        .into()
    }
}
