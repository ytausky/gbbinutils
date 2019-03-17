use self::value::Value;

use super::{BinaryObject, NameDef, Node, Program, RelocId, Section};

use crate::diag::BackendDiagnostics;

use std::borrow::Borrow;

mod eval;
mod translate;
mod value;

impl<S: Clone> Program<S> {
    pub(crate) fn link(&self, diagnostics: &mut impl BackendDiagnostics<S>) -> BinaryObject {
        let relocs = self.resolve_relocs();
        let mut context = EvalContext {
            program: self,
            relocs: &relocs,
            location: 0.into(),
        };
        BinaryObject {
            sections: self
                .sections
                .iter()
                .map(|section| section.translate(&mut context, diagnostics))
                .collect(),
        }
    }

    fn resolve_relocs(&self) -> RelocTable {
        let mut relocs = RelocTable::new(self.relocs);
        relocs.refine_all(self);
        relocs.refine_all(self);
        relocs
    }
}

struct EvalContext<'a, R, S> {
    program: &'a Program<S>,
    relocs: R,
    location: Value,
}

struct RelocTable(Vec<Value>);

impl RelocTable {
    fn new(relocs: usize) -> Self {
        Self(vec![Value::Unknown; relocs])
    }

    fn get(&self, RelocId(id): RelocId) -> Value {
        self.0[id].clone()
    }

    fn refine(&mut self, RelocId(id): RelocId, value: Value) -> bool {
        let stored_value = &mut self.0[id];
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

    fn refine_all<S: Clone>(&mut self, program: &Program<S>) -> i32 {
        let mut refinements = 0;
        let context = &mut EvalContext {
            program,
            relocs: self,
            location: Value::Unknown,
        };
        for section in &program.sections {
            let (_, size) = section.traverse(context, |item, context| {
                if let Node::Symbol((name, _), expr) = item {
                    let id = match context.program.names.get_name_def(*name).unwrap() {
                        NameDef::Value(id) => *id,
                    };
                    let value = expr.eval(context, &mut ignore_undefined);
                    refinements += context.relocs.refine(id, value) as i32
                }
            });
            refinements += context.relocs.refine(section.size, size) as i32
        }
        refinements
    }
}

impl<S: Clone> Section<S> {
    fn traverse<R, F>(&self, context: &mut EvalContext<R, S>, mut f: F) -> (Value, Value)
    where
        R: Borrow<RelocTable>,
        F: FnMut(&Node<S>, &mut EvalContext<R, S>),
    {
        let addr = self.eval_addr(context);
        let mut offset = Value::from(0);
        for item in &self.items {
            offset += &item.size(&context);
            context.location = &addr + &offset;
            f(item, context)
        }
        (addr, offset)
    }

    fn eval_addr<R: Borrow<RelocTable>>(&self, context: &EvalContext<R, S>) -> Value {
        self.addr
            .as_ref()
            .map(|expr| expr.eval(context, &mut ignore_undefined))
            .unwrap_or_else(|| 0.into())
    }
}

fn ignore_undefined<S>(_: &S) {}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::backend::{AllocName, Backend, PartialBackend};
    use crate::diag::IgnoreDiagnostics;
    use crate::expr::{BinaryOperator, ExprVariant};
    use crate::model::{Atom, Attr};
    use crate::program::{NameDef, NameTable, ProgramBuilder, RelocId, Section};

    #[test]
    fn resolve_origin_relative_to_previous_section() {
        let origin1 = 0x150;
        let skipped_bytes = 0x10;
        let object = Program {
            sections: vec![
                Section {
                    name: None,
                    addr: Some(origin1.into()),
                    size: RelocId(0),
                    items: vec![Node::Byte(0x42)],
                },
                Section {
                    name: None,
                    addr: Some(
                        ExprVariant::Binary(
                            BinaryOperator::Plus,
                            Box::new(Atom::LocationCounter.into()),
                            Box::new(skipped_bytes.into()),
                        )
                        .into(),
                    ),
                    size: RelocId(1),
                    items: vec![Node::Byte(0x43)],
                },
            ],
            names: NameTable::new(),
            relocs: 2,
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
        builder.define_symbol((symbol_id, ()), Atom::LocationCounter.into());
        let object = builder.into_object();
        let relocs = object.resolve_relocs();
        let value_id = match object.names.get_name_def(symbol_id).unwrap() {
            NameDef::Value(id) => *id,
        };
        assert_eq!(relocs.get(value_id), addr.into());
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
            let value = object.alloc_reloc();
            let items = &mut object.sections[0].items;
            items.push(Node::LdInlineAddr(0, Atom::Attr(name, Attr::Addr).into()));
            object.names.define_name(name, NameDef::Value(value));
            items.push(Node::Symbol((name, ()), Atom::LocationCounter.into()))
        })
    }

    fn assert_section_size(expected: impl Into<Value>, f: impl FnOnce(&mut Program<()>)) {
        let mut program = Program::new();
        program.add_section(None);
        f(&mut program);
        let relocs = program.resolve_relocs();
        assert_eq!(relocs.get(program.sections[0].size), expected.into())
    }
}
