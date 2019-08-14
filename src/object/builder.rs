use super::*;

use crate::analysis::backend::*;
use crate::diag::span::WithSpan;
use crate::model::{BinOp, ExprOp, FnCall, Item, ParamId};
use crate::BuiltinSymbols;

pub(crate) struct ProgramBuilder<'a, S> {
    context: LinkageContext<&'a mut Content<S>, &'a mut VarTable>,
    state: Option<BuilderState<S>>,
}

enum BuilderState<S> {
    AnonSectionPrelude { addr: Option<Const<S>> },
    Section(usize),
    SectionPrelude(usize),
}

impl<'a, S> ProgramBuilder<'a, S> {
    pub fn new(Object { program, vars }: &'a mut Object<S>) -> Self {
        Self {
            context: LinkageContext {
                program,
                vars,
                location: 0.into(),
            },
            state: Some(BuilderState::AnonSectionPrelude { addr: None }),
        }
    }

    fn push(&mut self, node: Node<S>) {
        self.current_section().items.push(node)
    }

    fn current_section(&mut self) -> &mut Section<S> {
        match self.state.take().unwrap() {
            BuilderState::AnonSectionPrelude { addr } => {
                self.add_section(None);
                let index = self.context.program.sections.len() - 1;
                self.state = Some(BuilderState::Section(index));
                let section = &mut self.context.program.sections[index];
                section.constraints.addr = addr;
                section
            }
            BuilderState::SectionPrelude(index) | BuilderState::Section(index) => {
                self.state = Some(BuilderState::Section(index));
                &mut self.context.program.sections[index]
            }
        }
    }

    fn add_section(&mut self, symbol: Option<ContentSymbol>) {
        self.context.program.add_section(
            symbol,
            self.context.vars.alloc(),
            self.context.vars.alloc(),
        )
    }
}

impl<'a, S: Clone> PartialBackend<S> for ProgramBuilder<'a, S> {
    type Value = Const<S>;

    fn emit_item(&mut self, item: Item<Self::Value>) {
        use super::lowering::Lower;
        item.lower().for_each(|data_item| self.push(data_item))
    }

    fn reserve(&mut self, bytes: Self::Value) {
        self.current_section().items.push(Node::Reserved(bytes))
    }

    fn set_origin(&mut self, addr: Self::Value) {
        match self.state.take().unwrap() {
            BuilderState::SectionPrelude(index) => {
                self.context.program.sections[index].constraints.addr = Some(addr);
                self.state = Some(BuilderState::SectionPrelude(index))
            }
            _ => self.state = Some(BuilderState::AnonSectionPrelude { addr: Some(addr) }),
        }
    }
}

impl<'a, S: Clone> Backend<S> for ProgramBuilder<'a, S> {
    type ConstBuilder = RelocContext<Self, Const<S>>;
    type SymbolBuilder = SymbolBuilder<'a, S>;

    fn build_const(self) -> Self::ConstBuilder {
        RelocContext::new(self)
    }

    fn define_symbol(self, name: Self::Name, span: S) -> Self::SymbolBuilder {
        let location = self.context.vars.alloc();
        SymbolBuilder {
            parent: self,
            location,
            name: (name.content().unwrap(), span),
            expr: Default::default(),
        }
    }
}

impl<'a, S: Clone> AllocName<S> for RelocContext<ProgramBuilder<'a, S>, Const<S>> {
    type Name = Symbol;

    fn alloc_name(&mut self, span: S) -> Self::Name {
        self.parent.alloc_name(span)
    }
}

impl<'a, S: Clone> Finish for RelocContext<ProgramBuilder<'a, S>, Const<S>> {
    type Parent = ProgramBuilder<'a, S>;
    type Value = Const<S>;

    fn finish(self) -> (Self::Parent, Self::Value) {
        (self.parent, self.builder)
    }
}

pub(crate) struct SymbolBuilder<'a, S> {
    parent: ProgramBuilder<'a, S>,
    location: VarId,
    name: (ContentSymbol, S),
    expr: Expr<S>,
}

impl<'a, S: Clone> AllocName<S> for SymbolBuilder<'a, S> {
    type Name = Symbol;

    fn alloc_name(&mut self, span: S) -> Self::Name {
        self.parent.alloc_name(span)
    }
}

macro_rules! impl_push_op_for_symbol_builder {
    ($t:ty) => {
        impl<'a, S: Clone> PushOp<$t, S> for SymbolBuilder<'a, S> {
            fn push_op(&mut self, op: $t, span: S) {
                self.expr.push_op(op, span)
            }
        }
    };
}

impl_push_op_for_symbol_builder! {i32}
impl_push_op_for_symbol_builder! {Name<Symbol>}
impl_push_op_for_symbol_builder! {ParamId}
impl_push_op_for_symbol_builder! {BinOp}
impl_push_op_for_symbol_builder! {FnCall}

impl<'a, S: Clone> PushOp<LocationCounter, S> for SymbolBuilder<'a, S> {
    fn push_op(&mut self, _: LocationCounter, span: S) {
        self.expr
            .0
            .push(ExprOp::Atom(Atom::Location(self.location)).with_span(span))
    }
}

impl<'a, S> Finish for SymbolBuilder<'a, S> {
    type Parent = ProgramBuilder<'a, S>;
    type Value = ();

    fn finish(self) -> (Self::Parent, Self::Value) {
        let mut parent = self.parent;
        parent.push(Node::Reloc(self.location));
        parent
            .context
            .program
            .symbols
            .define(self.name.0, ProgramDef::Expr(self.expr));
        (parent, ())
    }
}

impl<'a, S: Clone> AllocName<S> for ProgramBuilder<'a, S> {
    type Name = Symbol;

    fn alloc_name(&mut self, _span: S) -> Self::Name {
        self.context.program.symbols.alloc().into()
    }
}

impl<'a, S: Clone> StartSection<Symbol, S> for ProgramBuilder<'a, S> {
    fn start_section(&mut self, (name, _): (Symbol, S)) {
        let index = self.context.program.sections.len();
        self.state = Some(BuilderState::SectionPrelude(index));
        self.add_section(Some(name.content().unwrap()))
    }
}

impl<'a, S: Clone> BuiltinSymbols for ProgramBuilder<'a, S> {
    type Name = Symbol;

    fn builtin_symbols(&self) -> &[(&str, Self::Name)] {
        super::eval::BUILTIN_SYMBOLS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::diag::{CompactDiag, Message, TestDiagnosticsListener};
    use crate::link::BinaryObject;
    use crate::model::{BinOp, Instruction, Nullary, Width};
    use crate::object::SectionId;
    use std::borrow::Borrow;

    #[test]
    fn new_object_has_no_sections() {
        let object = build_object::<_, ()>(|_| ());
        assert_eq!(object.program.sections.len(), 0)
    }

    #[test]
    fn no_origin_by_default() {
        let object = build_object::<_, ()>(|mut builder| builder.push(Node::Byte(0xcd)));
        assert_eq!(object.program.sections[0].constraints.addr, None)
    }

    #[test]
    fn constrain_origin_determines_origin_of_new_section() {
        let origin: Const<_> = 0x3000.into();
        let object = build_object(|mut builder| {
            builder.set_origin(origin.clone());
            builder.push(Node::Byte(0xcd))
        });
        assert_eq!(object.program.sections[0].constraints.addr, Some(origin))
    }

    #[test]
    fn start_section_adds_named_section() {
        let mut wrapped_name = None;
        let object = build_object(|mut builder| {
            let name = builder.alloc_name(());
            builder.start_section((name, ()));
            wrapped_name = Some(name);
        });
        assert_eq!(
            object
                .program
                .symbols
                .get(wrapped_name.unwrap().content().unwrap()),
            Some(&ProgramDef::Section(SectionId(0)))
        )
    }

    #[test]
    fn set_origin_in_section_prelude_sets_origin() {
        let origin: Const<_> = 0x0150.into();
        let object = build_object(|mut builder| {
            let name = builder.alloc_name(());
            builder.start_section((name, ()));
            builder.set_origin(origin.clone())
        });
        assert_eq!(object.program.sections[0].constraints.addr, Some(origin))
    }

    #[test]
    fn push_node_into_named_section() {
        let node = Node::Byte(0x42);
        let object = build_object(|mut builder| {
            let name = builder.alloc_name(());
            builder.start_section((name, ()));
            builder.push(node.clone())
        });
        assert_eq!(object.program.sections[0].items, [node])
    }

    fn build_object<F: FnOnce(ProgramBuilder<S>), S>(f: F) -> Object<S> {
        let mut linkable = Object::new();
        let builder = ProgramBuilder::new(&mut linkable);
        f(builder);
        linkable
    }

    #[test]
    fn emit_stop() {
        emit_items_and_compare(
            [Item::Instruction(Instruction::Nullary(Nullary::Stop))],
            [0x10, 0x00],
        )
    }

    fn emit_items_and_compare<I, B>(items: I, bytes: B)
    where
        I: Borrow<[Item<Const<()>>]>,
        B: Borrow<[u8]>,
    {
        let (object, _) = with_object_builder(|mut builder| {
            for item in items.borrow() {
                builder.emit_item(item.clone())
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

    fn byte_literal(value: i32) -> Item<Const<()>> {
        Item::Data(value.into(), Width::Byte)
    }

    #[test]
    fn emit_diagnostic_when_byte_item_out_of_range() {
        test_diagnostic_for_out_of_range_byte(i32::from(i8::min_value()) - 1);
        test_diagnostic_for_out_of_range_byte(i32::from(u8::max_value()) + 1)
    }

    fn test_diagnostic_for_out_of_range_byte(value: i32) {
        let (_, diagnostics) =
            with_object_builder(|mut builder| builder.emit_item(byte_literal(value)));
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
        let (_, diagnostics) = with_object_builder(|mut builder| {
            let symbol_id = builder.alloc_name(name.into());
            let mut value: Const<_> = Default::default();
            value.push_op(symbol_id, name.into());
            builder.emit_item(word_item(value))
        });
        assert_eq!(*diagnostics, [unresolved(name)]);
    }

    #[test]
    fn diagnose_two_unresolved_symbols_in_one_expr() {
        let name1 = "ident1";
        let name2 = "ident2";
        let (_, diagnostics) = with_object_builder(|mut builder| {
            let value = {
                let id1 = builder.alloc_name(name1.into());
                let mut value: Const<_> = Default::default();
                value.push_op(id1, name1.into());
                let id2 = builder.alloc_name(name2.into());
                value.push_op(id2, name2.into());
                value.push_op(BinOp::Minus, "diff".into());
                value
            };
            builder.emit_item(word_item(value))
        });
        assert_eq!(*diagnostics, [unresolved(name1), unresolved(name2)]);
    }

    #[test]
    fn emit_defined_symbol() {
        let (object, diagnostics) = with_object_builder(|mut builder| {
            let symbol_id = builder.alloc_name(());
            let mut builder = builder.define_symbol(symbol_id, ());
            builder.push_op(LocationCounter, ());
            let (mut builder, _) = builder.finish();
            let mut value: Const<_> = Default::default();
            value.push_op(symbol_id, ());
            builder.emit_item(word_item(value));
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x00, 0x00])
    }

    #[test]
    fn emit_symbol_defined_after_use() {
        let (object, diagnostics) = with_object_builder(|mut builder| {
            let symbol_id = builder.alloc_name(());
            let mut value: Const<_> = Default::default();
            value.push_op(symbol_id, ());
            builder.emit_item(word_item(value));
            let mut builder = builder.define_symbol(symbol_id, ());
            builder.push_op(LocationCounter, ());
            builder.finish();
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x02, 0x00])
    }

    #[test]
    fn reserve_bytes_in_section() {
        let bytes = 3;
        let program = build_object(|mut builder| builder.reserve(bytes.into()));
        assert_eq!(
            program.program.sections[0].items,
            [Node::Reserved(bytes.into())]
        )
    }

    fn with_object_builder<S, F>(f: F) -> (BinaryObject, Box<[CompactDiag<S, S>]>)
    where
        S: Clone + 'static,
        F: FnOnce(ProgramBuilder<S>),
    {
        let mut diagnostics = TestDiagnosticsListener::new();
        let object = build_object(f).link(&mut diagnostics);
        let diagnostics = diagnostics.diagnostics.into_inner().into_boxed_slice();
        (object, diagnostics)
    }

    fn word_item<S: Clone>(value: Const<S>) -> Item<Const<S>> {
        Item::Data(value, Width::Word)
    }

    fn unresolved(symbol: impl Into<String>) -> CompactDiag<String, String> {
        let symbol = symbol.into();
        Message::UnresolvedSymbol {
            symbol: symbol.clone(),
        }
        .at(symbol)
        .into()
    }
}
