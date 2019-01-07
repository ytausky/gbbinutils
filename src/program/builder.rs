use super::{Chunk, NameId, Node, Program, Value};
use crate::backend::{
    Backend, BuildValue, HasValue, Item, RelocAtom, RelocExpr, RelocExprBuilder, ToValue,
};
use std::collections::HashMap;

pub struct ProgramBuilder<SR> {
    program: Program<SR>,
    state: Option<BuilderState<SR>>,
    names: HashMap<String, NameId>,
}

enum BuilderState<SR> {
    Pending {
        origin: Option<RelocExpr<NameId, SR>>,
    },
    InChunk(usize),
}

impl<SR> ProgramBuilder<SR> {
    pub fn new() -> ProgramBuilder<SR> {
        ProgramBuilder {
            program: Program::new(),
            state: Some(BuilderState::Pending { origin: None }),
            names: HashMap::new(),
        }
    }

    fn push(&mut self, node: Node<SR>) {
        self.current_chunk().items.push(node)
    }

    fn lookup(&mut self, name: String) -> NameId {
        let symbols = &mut self.program.symbols;
        *self.names.entry(name).or_insert_with(|| symbols.new_name())
    }

    fn current_chunk(&mut self) -> &mut Chunk<SR> {
        match self.state.take().unwrap() {
            BuilderState::Pending { origin } => {
                self.program.add_chunk();
                let index = self.program.chunks.len() - 1;
                self.state = Some(BuilderState::InChunk(index));
                let chunk = &mut self.program.chunks[index];
                chunk.origin = origin;
                chunk
            }
            BuilderState::InChunk(index) => {
                self.state = Some(BuilderState::InChunk(index));
                &mut self.program.chunks[index]
            }
        }
    }
}

impl<S: Clone + 'static> Backend<String, S> for ProgramBuilder<S> {
    type Object = Program<S>;

    fn define_symbol(&mut self, (name, span): (String, S), value: Self::Value) {
        let name_id = self.lookup(name);
        self.program.symbols.define_name(name_id, Value::Unknown);
        self.push(Node::Symbol((name_id, span), value))
    }

    fn emit_item(&mut self, item: Item<Self::Value>) {
        use super::lowering::Lower;
        item.lower().for_each(|data_item| self.push(data_item))
    }

    fn into_object(self) -> Self::Object {
        self.program
    }

    fn set_origin(&mut self, origin: Self::Value) {
        self.state = Some(BuilderState::Pending {
            origin: Some(origin),
        })
    }
}

impl<'a, S: Clone> HasValue<S> for RelocExprBuilder<&'a mut ProgramBuilder<S>> {
    type Value = RelocExpr<NameId, S>;
}

impl<'a, S: Clone> ToValue<String, S> for RelocExprBuilder<&'a mut ProgramBuilder<S>> {
    fn to_value(&mut self, (name, span): (String, S)) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::Symbol(self.0.lookup(name)), span)
    }
}

impl<S: Clone> HasValue<S> for ProgramBuilder<S> {
    type Value = RelocExpr<NameId, S>;
}

impl<'a, S: Clone + 'static> BuildValue<'a, String, S> for ProgramBuilder<S> {
    type Builder = RelocExprBuilder<&'a mut Self>;

    fn build_value(&'a mut self) -> Self::Builder {
        RelocExprBuilder(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::BinaryObject;
    use crate::backend::{ApplyBinaryOperator, Width};
    use crate::diag::{CompactDiagnostic, Message, TestDiagnosticsListener};
    use crate::expr::BinaryOperator;
    use crate::instruction::{Instruction, Nullary};
    use std::borrow::Borrow;

    #[test]
    fn new_object_has_no_chunks() {
        let object = build_object(|_| ());
        assert_eq!(object.chunks.len(), 0)
    }

    #[test]
    fn no_origin_by_default() {
        let object = build_object(|builder| builder.push(Node::Byte(0xcd)));
        assert_eq!(object.chunks[0].origin, None)
    }

    #[test]
    fn constrain_origin_determines_origin_of_new_chunk() {
        let origin: RelocExpr<_, _> = 0x3000.into();
        let object = build_object(|builder| {
            builder.set_origin(origin.clone());
            builder.push(Node::Byte(0xcd))
        });
        assert_eq!(object.chunks[0].origin, Some(origin))
    }

    fn build_object(f: impl FnOnce(&mut ProgramBuilder<()>)) -> Program<()> {
        let mut builder = ProgramBuilder::new();
        f(&mut builder);
        builder.into_object()
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
        I: Borrow<[Item<RelocExpr<NameId, ()>>]>,
        B: Borrow<[u8]>,
    {
        let (object, _) = with_object_builder(|builder| {
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

    fn byte_literal(value: i32) -> Item<RelocExpr<NameId, ()>> {
        Item::Data(value.into(), Width::Byte)
    }

    #[test]
    fn emit_diagnostic_when_byte_item_out_of_range() {
        test_diagnostic_for_out_of_range_byte(i8::min_value() as i32 - 1);
        test_diagnostic_for_out_of_range_byte(u8::max_value() as i32 + 1)
    }

    fn test_diagnostic_for_out_of_range_byte(value: i32) {
        let (_, diagnostics) =
            with_object_builder(|builder| builder.emit_item(byte_literal(value)));
        assert_eq!(
            *diagnostics,
            [CompactDiagnostic::new(
                Message::ValueOutOfRange {
                    value,
                    width: Width::Byte,
                },
                ()
            )]
        );
    }

    #[test]
    fn diagnose_unresolved_symbol() {
        let ident = "ident";
        let (_, diagnostics) = with_object_builder(|builder| {
            let value = builder
                .build_value()
                .to_value((ident.to_string(), ident.into()));
            builder.emit_item(word_item(value))
        });
        assert_eq!(*diagnostics, [unresolved(ident)]);
    }

    #[test]
    fn diagnose_two_unresolved_symbols_in_one_expr() {
        let ident1 = "ident1";
        let ident2 = "ident2";
        let (_, diagnostics) = with_object_builder(|builder| {
            let value = {
                let mut builder = builder.build_value();
                let lhs = builder.to_value((ident1.to_string(), ident1.into()));
                let rhs = builder.to_value((ident2.to_string(), ident2.into()));
                builder.apply_binary_operator((BinaryOperator::Minus, "diff".into()), lhs, rhs)
            };
            builder.emit_item(word_item(value))
        });
        assert_eq!(*diagnostics, [unresolved(ident1), unresolved(ident2)]);
    }

    #[test]
    fn emit_defined_symbol() {
        let label = "label";
        let (object, diagnostics) = with_object_builder(|builder| {
            builder.define_symbol((label.into(), ()), RelocAtom::LocationCounter.into());
            let value = builder.build_value().to_value((label.to_string(), ()));
            builder.emit_item(word_item(value));
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x00, 0x00])
    }

    #[test]
    fn emit_symbol_defined_after_use() {
        let label = "label";
        let (object, diagnostics) = with_object_builder(|builder| {
            let value = builder.build_value().to_value((label.to_string(), ()));
            builder.emit_item(word_item(value));
            builder.define_symbol((label.into(), ()), RelocAtom::LocationCounter.into());
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x02, 0x00])
    }

    fn with_object_builder<S: Clone + 'static, F: FnOnce(&mut ProgramBuilder<S>)>(
        f: F,
    ) -> (BinaryObject, Box<[CompactDiagnostic<S, S>]>) {
        let mut diagnostics = TestDiagnosticsListener::new();
        let object = {
            let mut builder = ProgramBuilder::new();
            f(&mut builder);
            builder.into_object().link(&mut diagnostics)
        };
        let diagnostics = diagnostics.diagnostics.into_inner().into_boxed_slice();
        (object, diagnostics)
    }

    fn word_item<S: Clone>(value: RelocExpr<NameId, S>) -> Item<RelocExpr<NameId, S>> {
        Item::Data(value, Width::Word)
    }

    fn unresolved(symbol: impl Into<String>) -> CompactDiagnostic<String, String> {
        let symbol = symbol.into();
        CompactDiagnostic::new(
            Message::UnresolvedSymbol {
                symbol: symbol.clone(),
            },
            symbol,
        )
    }
}
