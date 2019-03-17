use self::value::Value;

use super::{BinaryObject, NameDef, NameTable, Node, Program, Section, ValueId};

use crate::diag::BackendDiagnostics;

use std::borrow::Borrow;

mod eval;
mod translate;
mod value;

impl<S: Clone> Program<S> {
    pub(crate) fn link(mut self, diagnostics: &mut impl BackendDiagnostics<S>) -> BinaryObject {
        self.resolve_relocs();
        let mut context = EvalContext {
            names: &self.names,
            relocs: &self.relocs,
            location: 0.into(),
        };
        BinaryObject {
            sections: self
                .sections
                .into_iter()
                .map(|section| section.translate(&mut context, diagnostics))
                .collect(),
        }
    }

    fn resolve_relocs(&mut self) {
        self.refine_symbols();
        self.refine_symbols();
    }

    fn refine_symbols(&mut self) -> i32 {
        let mut refinements = 0;
        let context = &mut EvalContext {
            names: &self.names,
            relocs: &mut self.relocs,
            location: Value::Unknown,
        };
        for section in &self.sections {
            let (_, size) = section.traverse(context, |item, context| {
                if let Node::Symbol((name, _), expr) = item {
                    let id = match context.names.get_name_def(*name).unwrap() {
                        NameDef::Value(id) => *id,
                    };
                    let value = expr.evaluate(context);
                    refinements += context.relocs.refine(id, value) as i32
                }
            });
            refinements += context.relocs.refine(section.size, size) as i32
        }
        refinements
    }
}

pub(super) struct EvalContext<'a, R> {
    pub names: &'a NameTable,
    pub relocs: R,
    pub location: Value,
}

pub(super) struct RelocTable {
    values: Vec<Value>,
}

impl RelocTable {
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    pub fn alloc(&mut self) -> ValueId {
        let id = ValueId(self.values.len());
        self.values.push(Value::Unknown);
        id
    }

    pub(super) fn get_value(&self, ValueId(id): ValueId) -> Value {
        self.values[id].clone()
    }

    pub fn refine(&mut self, ValueId(id): ValueId, value: Value) -> bool {
        let stored_value = &mut self.values[id];
        let old_value = stored_value.clone();
        let was_refined = match (old_value, &value) {
            (Value::Unknown, new_value) => *new_value != Value::Unknown,
            (
                Value::Range {
                    min: old_min,
                    max: old_max,
                },
                Value::Range {
                    min: new_min,
                    max: new_max,
                },
            ) => {
                assert!(*new_min >= old_min);
                assert!(*new_max <= old_max);
                *new_min > old_min || *new_max < old_max
            }
            (Value::Range { .. }, Value::Unknown) => {
                panic!("a symbol previously approximated is now unknown")
            }
        };
        *stored_value = value;
        was_refined
    }
}

impl<S: Clone> Section<S> {
    fn traverse<R, F>(&self, context: &mut EvalContext<R>, mut f: F) -> (Value, Value)
    where
        R: Borrow<RelocTable>,
        F: FnMut(&Node<S>, &mut EvalContext<R>),
    {
        let addr = self.evaluate_addr(context);
        let mut offset = Value::from(0);
        for item in &self.items {
            offset += &item.size(&context);
            context.location = &addr + &offset;
            f(item, context)
        }
        (addr, offset)
    }

    fn evaluate_addr<R: Borrow<RelocTable>>(&self, context: &EvalContext<R>) -> Value {
        self.addr
            .as_ref()
            .map(|expr| expr.evaluate(context))
            .unwrap_or_else(|| 0.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::backend::{AllocName, Backend, PartialBackend};
    use crate::diag::IgnoreDiagnostics;
    use crate::expr::{BinaryOperator, ExprVariant};
    use crate::model::RelocAtom;
    use crate::program::{NameDef, ProgramBuilder, Section, ValueId};

    #[test]
    fn resolve_origin_relative_to_previous_section() {
        let origin1 = 0x150;
        let skipped_bytes = 0x10;
        let object = Program {
            sections: vec![
                Section {
                    name: None,
                    addr: Some(origin1.into()),
                    size: ValueId(0),
                    items: vec![Node::Byte(0x42)],
                },
                Section {
                    name: None,
                    addr: Some(
                        ExprVariant::Binary(
                            BinaryOperator::Plus,
                            Box::new(RelocAtom::LocationCounter.into()),
                            Box::new(skipped_bytes.into()),
                        )
                        .into(),
                    ),
                    size: ValueId(1),
                    items: vec![Node::Byte(0x43)],
                },
            ],
            names: NameTable::new(),
            relocs: {
                let mut table = RelocTable::new();
                table.alloc();
                table.alloc();
                table
            },
        };
        let binary = object.link(&mut IgnoreDiagnostics::new());
        assert_eq!(
            binary.sections[1].addr,
            (origin1 + 1 + skipped_bytes) as usize
        )
    }

    #[test]
    fn label_defined_as_section_origin_plus_offset() {
        let addr = 0xffe1;
        let mut builder = ProgramBuilder::new();
        builder.set_origin(addr.into());
        let symbol_id = builder.alloc_name(());
        builder.define_symbol((symbol_id, ()), RelocAtom::LocationCounter.into());
        let mut object = builder.into_object();
        object.resolve_relocs();
        let value_id = match object.names.get_name_def(symbol_id).unwrap() {
            NameDef::Value(id) => *id,
        };
        assert_eq!(object.relocs.get_value(value_id), addr.into());
    }

    #[test]
    fn empty_section_has_size_zero() {
        assert_section_size(0, |_| ())
    }

    #[test]
    fn section_with_one_byte_has_size_one() {
        assert_section_size(1, |object| object.sections[0].items.push(Node::Byte(0x42)));
    }

    #[test]
    fn section_with_const_inline_addr_ld_has_size_two() {
        test_section_size_with_literal_ld_inline_addr(0xff00, 2)
    }

    #[test]
    fn section_with_const_inline_addr_ld_has_size_three() {
        test_section_size_with_literal_ld_inline_addr(0x1234, 3)
    }

    fn test_section_size_with_literal_ld_inline_addr(addr: i32, expected: i32) {
        assert_section_size(expected, |object| {
            object.sections[0]
                .items
                .push(Node::LdInlineAddr(0, addr.into()))
        });
    }

    #[test]
    fn ld_inline_addr_with_symbol_after_instruction_has_size_three() {
        assert_section_size(3, |object| {
            let name = object.names.alloc_name();
            let value = object.relocs.alloc();
            let items = &mut object.sections[0].items;
            items.push(Node::LdInlineAddr(0, RelocAtom::Name(name).into()));
            object.names.define_name(name, NameDef::Value(value));
            items.push(Node::Symbol((name, ()), RelocAtom::LocationCounter.into()))
        })
    }

    fn assert_section_size(expected: impl Into<Value>, f: impl FnOnce(&mut Program<()>)) {
        let mut program = Program::new();
        program.add_section(None);
        f(&mut program);
        program.resolve_relocs();
        assert_eq!(
            program.relocs.get_value(program.sections[0].size),
            expected.into()
        )
    }
}
