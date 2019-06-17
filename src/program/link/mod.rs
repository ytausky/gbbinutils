pub use self::eval::BUILTIN_NAMES;

use self::num::Num;

use super::{BinaryObject, Node, Program, RelocId, Section};

use crate::diag::{BackendDiagnostics, IgnoreDiagnostics};
use crate::model::Width;

use std::borrow::Borrow;

mod eval;
mod num;
mod translate;

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
                .flat_map(|section| section.translate(&mut context, diagnostics))
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
    location: Num,
}

struct RelocTable(Vec<Num>);

impl RelocTable {
    fn new(relocs: usize) -> Self {
        Self(vec![Num::Unknown; relocs])
    }

    fn get(&self, RelocId(id): RelocId) -> Num {
        self.0[id].clone()
    }

    fn refine(&mut self, RelocId(id): RelocId, value: Num) -> bool {
        let stored_value = &mut self.0[id];
        let old_value = stored_value.clone();
        let was_refined = match (old_value, &value) {
            (Num::Unknown, new_value) => *new_value != Num::Unknown,
            (
                Num::Range {
                    min: old_min,
                    max: old_max,
                },
                Num::Range {
                    min: new_min,
                    max: new_max,
                },
            ) => {
                assert!(*new_min >= old_min);
                assert!(*new_max <= old_max);
                *new_min > old_min || *new_max < old_max
            }
            (Num::Range { .. }, Num::Unknown) => {
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
            location: Num::Unknown,
        };
        for section in &program.sections {
            context.location = section.eval_addr(&context);
            context
                .relocs
                .refine(section.addr, context.location.clone());
            let size = section.traverse(context, |item, context| {
                if let Node::Reloc(id) = item {
                    refinements += context.relocs.refine(*id, context.location.clone()) as i32
                }
            });
            refinements += context.relocs.refine(section.size, size) as i32
        }
        refinements
    }
}

impl<S: Clone> Section<S> {
    fn traverse<R, F>(&self, context: &mut EvalContext<R, S>, mut f: F) -> Num
    where
        R: Borrow<RelocTable>,
        F: FnMut(&Node<S>, &mut EvalContext<R, S>),
    {
        let addr = context.location.clone();
        let mut offset = Num::from(0);
        for item in &self.items {
            offset += &item.size(&context);
            context.location = &addr + &offset;
            f(item, context)
        }
        offset
    }

    fn eval_addr<R: Borrow<RelocTable>>(&self, context: &EvalContext<R, S>) -> Num {
        self.constraints
            .addr
            .as_ref()
            .map(|expr| expr.eval(context, &mut IgnoreDiagnostics))
            .unwrap_or_else(|| 0.into())
    }
}

impl<S: Clone> Node<S> {
    fn size<R: Borrow<RelocTable>>(&self, context: &EvalContext<R, S>) -> Num {
        match self {
            Node::Byte(_) | Node::Embedded(..) => 1.into(),
            Node::Immediate(_, width) => width.len().into(),
            Node::LdInlineAddr(_, expr) => match expr.eval(context, &mut IgnoreDiagnostics) {
                Num::Range { min, .. } if min >= 0xff00 => 2.into(),
                Num::Range { max, .. } if max < 0xff00 => 3.into(),
                _ => Num::Range { min: 2, max: 3 },
            },
            Node::Reloc(_) => 0.into(),
            Node::Reserved(bytes) => bytes.eval(context, &mut IgnoreDiagnostics),
        }
    }
}

impl Width {
    fn len(self) -> i32 {
        match self {
            Width::Byte => 1,
            Width::Word => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::backend::*;
    use crate::diag::IgnoreDiagnostics;
    use crate::model::{BinOp, Width};
    use crate::program::*;

    #[test]
    fn resolve_origin_relative_to_previous_section() {
        let origin1 = 0x150;
        let skipped_bytes = 0x10;
        let object = Program {
            sections: vec![
                Section {
                    constraints: Constraints {
                        addr: Some(origin1.into()),
                    },
                    addr: RelocId(0),
                    size: RelocId(1),
                    items: vec![Node::Byte(0x42)],
                },
                Section {
                    constraints: Constraints {
                        addr: Some(Immediate::from_items(&[
                            LocationCounter.into(),
                            skipped_bytes.into(),
                            BinOp::Plus.into(),
                        ])),
                    },
                    addr: RelocId(2),
                    size: RelocId(3),
                    items: vec![Node::Byte(0x43)],
                },
            ],
            names: NameTable::new(),
            relocs: 4,
        };
        let binary = object.link(&mut IgnoreDiagnostics);
        assert_eq!(
            binary.sections[1].addr,
            (origin1 + 1 + skipped_bytes) as usize
        )
    }

    #[test]
    fn label_defined_as_section_origin_plus_offset() {
        let addr = 0xffe1;
        let mut program = Program::new();
        let mut builder = ProgramBuilder::new(&mut program);
        builder.set_origin(addr.into());
        let symbol_id = builder.alloc_name(());
        let mut builder = builder.define_symbol(symbol_id, ());
        builder.push_op(LocationCounter, ());
        builder.finish_fn_def();
        let relocs = program.resolve_relocs();
        assert_eq!(relocs.get(RelocId(0)), addr.into());
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
            let name = object.names.alloc();
            let reloc = object.alloc_reloc();
            let items = &mut object.sections[0].items;
            items.push(Node::LdInlineAddr(0, Atom::Name(name.into()).into()));
            items.push(Node::Reloc(reloc));
            object
                .names
                .define(name, NameDef::Symbol(Atom::Location(reloc).into()))
        })
    }

    #[test]
    fn resolve_expr_with_section_addr() {
        let program = Program {
            sections: vec![Section {
                constraints: Constraints {
                    addr: Some(0x1337.into()),
                },
                addr: RelocId(0),
                size: RelocId(1),
                items: vec![Node::Immediate(
                    Atom::Name(NameDefId(0).into()).into(),
                    Width::Word,
                )],
            }],
            names: NameTable(vec![Some(NameDef::Section(SectionId(0)))]),
            relocs: 2,
        };
        let binary = program.link(&mut IgnoreDiagnostics);
        assert_eq!(binary.sections[0].data, [0x37, 0x13])
    }

    #[test]
    fn traverse_reserved_bytes() {
        let addr = 0x0100;
        let bytes = 10;
        let symbol = RelocId(2);
        let program = Program::<()> {
            sections: vec![Section {
                constraints: Constraints {
                    addr: Some(addr.into()),
                },
                addr: RelocId(0),
                size: RelocId(1),
                items: vec![
                    Node::Reserved(bytes.into()),
                    Node::Reloc(symbol),
                    Node::Immediate(Atom::Name(NameDefId(0).into()).into(), Width::Word),
                ],
            }],
            names: NameTable(vec![Some(NameDef::Symbol(Atom::Location(symbol).into()))]),
            relocs: 3,
        };
        let relocs = program.resolve_relocs();
        assert_eq!(relocs.get(symbol), (addr + bytes).into())
    }

    fn assert_section_size(expected: impl Into<Num>, f: impl FnOnce(&mut Program<()>)) {
        let mut program = Program::new();
        program.add_section(None);
        f(&mut program);
        let relocs = program.resolve_relocs();
        assert_eq!(relocs.get(program.sections[0].size), expected.into())
    }
}
