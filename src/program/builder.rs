use super::{Chunk, NameId, Node, Program, RelocExpr, Value};
use crate::backend::{
    ApplyBinaryOperator, Backend, BuildValue, HasValue, Item, Name, NameTable, PartialBackend,
    RelocAtom, RelocExprBuilder, ToValue, ValueFromIdent, ValueFromSimple,
};
use crate::expr::{BinaryOperator, Expr, ExprVariant};
use crate::frontend::Ident;

pub struct ProgramBuilder<SR> {
    program: Program<SR>,
    state: Option<BuilderState<SR>>,
}

enum BuilderState<S> {
    Pending { origin: Option<RelocExpr<S>> },
    InChunk(usize),
}

impl<SR> ProgramBuilder<SR> {
    pub fn new() -> ProgramBuilder<SR> {
        ProgramBuilder {
            program: Program::new(),
            state: Some(BuilderState::Pending { origin: None }),
        }
    }

    pub fn into_object(self) -> Program<SR> {
        self.program
    }

    fn push(&mut self, node: Node<SR>) {
        self.current_chunk().items.push(node)
    }

    fn lookup<N>(&mut self, name: Ident<String>, table: &mut N) -> NameId
    where
        N: NameTable<Ident<String>>,
    {
        match table.get(&name) {
            None => {
                let id = self.program.symbols.new_name();
                table.insert(name, Name::Symbol(id));
                id
            }
            Some(Name::Macro(_)) => unimplemented!(),
            Some(Name::Symbol(id)) => *id,
        }
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

impl<S: Clone> PartialBackend<S> for ProgramBuilder<S> {
    fn emit_item(&mut self, item: Item<Self::Value>) {
        use super::lowering::Lower;
        item.lower().for_each(|data_item| self.push(data_item))
    }

    fn set_origin(&mut self, origin: Self::Value) {
        self.state = Some(BuilderState::Pending {
            origin: Some(origin),
        })
    }
}

impl<S, N> Backend<Ident<String>, S, N> for ProgramBuilder<S>
where
    S: Clone + 'static,
    N: NameTable<Ident<String>> + 'static,
{
    fn define_symbol(
        &mut self,
        (ident, span): (Ident<String>, S),
        value: Self::Value,
        table: &mut N,
    ) {
        let name_id = self.lookup(ident, table);
        self.program.symbols.define_name(name_id, Value::Unknown);
        self.push(Node::Symbol((name_id, span), value))
    }
}

impl<S: Clone> ValueFromSimple<S> for ProgramBuilder<S> {
    fn from_location_counter(&mut self, span: S) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::LocationCounter, span)
    }

    fn from_number(&mut self, n: i32, span: S) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::Literal(n), span)
    }
}

impl<N, S> ValueFromIdent<N, Ident<String>, S> for ProgramBuilder<S>
where
    N: NameTable<Ident<String>>,
    S: Clone,
{
    fn from_ident(&mut self, ident: Ident<String>, span: S, names: &mut N) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::Symbol(self.lookup(ident, names)), span)
    }
}

impl<S: Clone> ApplyBinaryOperator<S> for ProgramBuilder<S> {
    fn apply_binary_operator(
        &mut self,
        operator: (BinaryOperator, S),
        left: Self::Value,
        right: Self::Value,
    ) -> Self::Value {
        Expr {
            variant: ExprVariant::Binary(operator.0, Box::new(left), Box::new(right)),
            span: operator.1,
        }
    }
}

impl<'a, M, S: Clone> HasValue<S> for RelocExprBuilder<'a, &'a mut ProgramBuilder<S>, M> {
    type Value = RelocExpr<S>;
}

impl<'a, N, S> ToValue<Ident<String>, S> for RelocExprBuilder<'a, &'a mut ProgramBuilder<S>, N>
where
    N: NameTable<Ident<String>>,
    S: Clone,
{
    fn to_value(&mut self, (ident, span): (Ident<String>, S)) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::Symbol(self.0.lookup(ident, self.1)), span)
    }
}

impl<S: Clone> HasValue<S> for ProgramBuilder<S> {
    type Value = RelocExpr<S>;
}

impl<'a, N, S> BuildValue<'a, Ident<String>, N, S> for ProgramBuilder<S>
where
    N: NameTable<Ident<String>> + 'a,
    S: Clone + 'static,
{
    type Builder = RelocExprBuilder<'a, &'a mut Self, N>;

    fn build_value(&'a mut self, table: &'a mut N) -> Self::Builder {
        RelocExprBuilder(self, table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::{ApplyBinaryOperator, BinaryObject, Width};
    use crate::diag::{CompactDiagnostic, Message, TestDiagnosticsListener};
    use crate::expr::BinaryOperator;
    use crate::instruction::{Instruction, Nullary};
    use std::borrow::Borrow;

    type NameTable = crate::backend::HashMapNameTable<()>;

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
        let origin: RelocExpr<_> = 0x3000.into();
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
        I: Borrow<[Item<RelocExpr<()>>]>,
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

    fn byte_literal(value: i32) -> Item<RelocExpr<()>> {
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
        let name = "ident";
        let ident = Ident { name: name.into() };
        let (_, diagnostics) = with_object_builder(|builder| {
            let value = builder
                .build_value(&mut NameTable::new())
                .to_value((ident, name.into()));
            builder.emit_item(word_item(value))
        });
        assert_eq!(*diagnostics, [unresolved(name)]);
    }

    #[test]
    fn diagnose_two_unresolved_symbols_in_one_expr() {
        let name1 = "ident1";
        let name2 = "ident2";
        let ident1 = Ident {
            name: name1.to_string(),
        };
        let ident2 = Ident {
            name: name2.to_string(),
        };
        let (_, diagnostics) = with_object_builder(|builder| {
            let names = &mut NameTable::new();
            let value = {
                let mut builder = builder.build_value(names);
                let lhs = builder.to_value((ident1, name1.into()));
                let rhs = builder.to_value((ident2, name2.into()));
                builder.apply_binary_operator((BinaryOperator::Minus, "diff".into()), lhs, rhs)
            };
            builder.emit_item(word_item(value))
        });
        assert_eq!(*diagnostics, [unresolved(name1), unresolved(name2)]);
    }

    #[test]
    fn emit_defined_symbol() {
        let label = "label";
        let ident = Ident { name: label.into() };
        let (object, diagnostics) = with_object_builder(|builder| {
            let mut table = NameTable::new();
            builder.define_symbol(
                (ident.clone(), ()),
                RelocAtom::LocationCounter.into(),
                &mut table,
            );
            let value = builder.build_value(&mut table).to_value((ident, ()));
            builder.emit_item(word_item(value));
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x00, 0x00])
    }

    #[test]
    fn emit_symbol_defined_after_use() {
        let label = "label";
        let (object, diagnostics) = with_object_builder(|builder| {
            let mut table = NameTable::new();
            let value = builder
                .build_value(&mut table)
                .to_value((Ident::from(label), ()));
            builder.emit_item(word_item(value));
            builder.define_symbol(
                (label.into(), ()),
                RelocAtom::LocationCounter.into(),
                &mut table,
            );
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

    fn word_item<S: Clone>(value: RelocExpr<S>) -> Item<RelocExpr<S>> {
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
