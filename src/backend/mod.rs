pub use backend::object::link;
pub use backend::object::ObjectBuilder;

use backend::{
    lowering::Lower, object::{Chunk, Node, Object},
};
use diagnostics::*;
use instruction::Instruction;
use Width;

mod lowering;
mod object;

pub trait Backend<R> {
    type Object;
    fn add_label(&mut self, label: (impl Into<String>, R));
    fn emit_item(&mut self, item: Item<R>);
    fn into_object(self) -> Self::Object;
    fn set_origin(&mut self, origin: RelocExpr<R>);
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item<R> {
    Data(RelocExpr<R>, Width),
    Instruction(Instruction<R>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum RelocExpr<SR> {
    Literal(i32, SR),
    LocationCounter(SR),
    Subtract(Box<RelocExpr<SR>>, Box<RelocExpr<SR>>, SR),
    Symbol(String, SR),
}

impl<SR: SourceRange> Source for RelocExpr<SR> {
    type Range = SR;
    fn source_range(&self) -> Self::Range {
        use backend::RelocExpr::*;
        match self {
            Literal(_, range)
            | Symbol(_, range)
            | LocationCounter(range)
            | Subtract(_, _, range) => (*range).clone(),
        }
    }
}

pub struct BinaryObject {
    sections: Vec<BinarySection>,
}

impl BinaryObject {
    pub fn into_rom(self) -> Rom {
        let mut data: Vec<u8> = Vec::new();
        self.sections
            .into_iter()
            .for_each(|section| data.extend(section.data.into_iter()));
        Rom {
            data: data.into_boxed_slice(),
        }
    }
}

pub struct Rom {
    pub data: Box<[u8]>,
}

impl<SR: SourceRange> Backend<SR> for ObjectBuilder<SR> {
    type Object = Object<SR>;

    fn add_label(&mut self, label: (impl Into<String>, SR)) {
        self.push(Node::Label(label.0.into(), label.1))
    }

    fn emit_item(&mut self, item: Item<SR>) {
        item.lower().for_each(|data_item| self.push(data_item))
    }

    fn into_object(self) -> Self::Object {
        self.build()
    }

    fn set_origin(&mut self, origin: RelocExpr<SR>) {
        self.constrain_origin(origin)
    }
}

struct BinarySection {
    origin: i32,
    data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use diagnostics::TestDiagnosticsListener;
    use instruction::Nullary;
    use std::borrow::Borrow;

    #[test]
    fn emit_literal_byte_item() {
        emit_items_and_compare([byte_literal(0xff)], [0xff])
    }

    #[test]
    fn emit_two_literal_byte_item() {
        emit_items_and_compare([byte_literal(0x12), byte_literal(0x34)], [0x12, 0x34])
    }

    fn byte_literal(value: i32) -> Item<()> {
        Item::Data(RelocExpr::Literal(value, ()), Width::Byte)
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
        I: Borrow<[Item<()>]>,
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
    fn emit_diagnostic_when_byte_item_out_of_range() {
        test_diagnostic_for_out_of_range_byte(i8::min_value() as i32 - 1);
        test_diagnostic_for_out_of_range_byte(u8::max_value() as i32 + 1)
    }

    fn test_diagnostic_for_out_of_range_byte(value: i32) {
        let (_, diagnostics) =
            with_object_builder(|builder| builder.emit_item(byte_literal(value)));
        assert_eq!(
            *diagnostics,
            [Diagnostic::new(
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
        let (_, diagnostics) =
            with_object_builder(|builder| builder.emit_item(symbol_expr_item(ident)));
        assert_eq!(*diagnostics, [unresolved(ident)]);
    }

    #[test]
    fn diagnose_two_unresolved_symbols_in_one_expr() {
        let ident1 = "ident1";
        let ident2 = "ident2";
        let (_, diagnostics) = with_object_builder(|builder| {
            builder.emit_item(Item::Data(
                RelocExpr::Subtract(
                    Box::new(symbol_expr(ident1)),
                    Box::new(symbol_expr(ident2)),
                    (),
                ),
                Width::Word,
            ))
        });
        assert_eq!(*diagnostics, [unresolved(ident1), unresolved(ident2)]);
    }

    #[test]
    fn emit_defined_symbol() {
        let label = "label";
        let (object, diagnostics) = with_object_builder(|builder| {
            builder.add_label((label, ()));
            builder.emit_item(symbol_expr_item(label));
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x00, 0x00])
    }

    #[test]
    fn emit_symbol_defined_after_use() {
        let label = "label";
        let (object, diagnostics) = with_object_builder(|builder| {
            builder.emit_item(symbol_expr_item(label));
            builder.add_label((label, ()));
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x02, 0x00])
    }

    type TestObjectBuilder = ObjectBuilder<()>;

    fn with_object_builder<F: FnOnce(&mut TestObjectBuilder)>(
        f: F,
    ) -> (BinaryObject, Box<[Diagnostic<()>]>) {
        let diagnostics = TestDiagnosticsListener::new();
        let object = {
            let mut builder = ObjectBuilder::new();
            f(&mut builder);
            link(builder.build(), &diagnostics)
        };
        let diagnostics = diagnostics.diagnostics.into_inner().into_boxed_slice();
        (object, diagnostics)
    }

    fn symbol_expr_item(symbol: impl Into<String>) -> Item<()> {
        Item::Data(symbol_expr(symbol), Width::Word)
    }

    fn symbol_expr(symbol: impl Into<String>) -> RelocExpr<()> {
        RelocExpr::Symbol(symbol.into(), ())
    }

    fn unresolved(symbol: impl Into<String>) -> Diagnostic<()> {
        Diagnostic::new(
            Message::UnresolvedSymbol {
                symbol: symbol.into(),
            },
            (),
        )
    }
}
