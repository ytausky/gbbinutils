pub use self::eval::BUILTIN_SYMBOLS;

use self::num::Num;

use super::{BinaryObject, Node, Program, Section, VarId};

use crate::diag::{BackendDiagnostics, IgnoreDiagnostics};
use crate::model::Width;

use std::borrow::Borrow;
use std::ops::{Index, IndexMut};

mod eval;
mod num;
mod translate;

impl<S: Clone> Program<S> {
    pub(crate) fn link(&self, diagnostics: &mut impl BackendDiagnostics<S>) -> BinaryObject {
        let vars = &self.resolve_relocs();
        let mut context = LinkageContext {
            program: self,
            vars,
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

    fn resolve_relocs(&self) -> VarTable {
        let mut relocs = VarTable(vec![Default::default(); self.link_vars]);
        relocs.refine_all(self);
        relocs.refine_all(self);
        relocs
    }
}

pub(super) struct LinkageContext<P, V> {
    pub program: P,
    pub vars: V,
    pub location: Num,
}

pub(super) struct VarTable(Vec<Var>);

#[derive(Clone, Default)]
pub(super) struct Var {
    value: Num,
}

impl Var {
    fn refine(&mut self, value: Num) -> bool {
        let old_value = self.value.clone();
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
        self.value = value;
        was_refined
    }
}

impl VarTable {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    fn refine_all<S: Clone>(&mut self, program: &Program<S>) -> i32 {
        let mut refinements = 0;
        let context = &mut LinkageContext {
            program,
            vars: self,
            location: Num::Unknown,
        };
        for section in &program.sections {
            context.location = section.eval_addr(context);
            context.vars[section.addr].refine(context.location.clone());
            let size = section.traverse(context, |item, context| {
                if let Node::Reloc(id) = item {
                    refinements += context.vars[*id].refine(context.location.clone()) as i32
                }
            });
            refinements += context.vars[section.size].refine(size) as i32
        }
        refinements
    }
}

impl Index<VarId> for VarTable {
    type Output = Var;

    fn index(&self, VarId(id): VarId) -> &Self::Output {
        &self.0[id]
    }
}

impl IndexMut<VarId> for VarTable {
    fn index_mut(&mut self, VarId(id): VarId) -> &mut Self::Output {
        &mut self.0[id]
    }
}

impl<S: Clone> Section<S> {
    fn traverse<V, F>(&self, context: &mut LinkageContext<&Program<S>, V>, mut f: F) -> Num
    where
        V: Borrow<VarTable>,
        F: FnMut(&Node<S>, &mut LinkageContext<&Program<S>, V>),
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

    fn eval_addr<'a, V: Borrow<VarTable>>(
        &self,
        context: &LinkageContext<&'a Program<S>, V>,
    ) -> Num {
        self.constraints
            .addr
            .as_ref()
            .map(|expr| expr.to_num(context, &mut IgnoreDiagnostics))
            .unwrap_or_else(|| 0.into())
    }
}

impl<S: Clone> Node<S> {
    fn size<'a, V: Borrow<VarTable>>(&self, context: &LinkageContext<&'a Program<S>, V>) -> Num {
        match self {
            Node::Byte(_) | Node::Embedded(..) => 1.into(),
            Node::Immediate(_, width) => width.len().into(),
            Node::LdInlineAddr(_, expr) => match expr.to_num(context, &mut IgnoreDiagnostics) {
                Num::Range { min, .. } if min >= 0xff00 => 2.into(),
                Num::Range { max, .. } if max < 0xff00 => 3.into(),
                _ => Num::Range { min: 2, max: 3 },
            },
            Node::Reloc(_) => 0.into(),
            Node::Reserved(bytes) => bytes.to_num(context, &mut IgnoreDiagnostics),
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
                    addr: VarId(0),
                    size: VarId(1),
                    items: vec![Node::Byte(0x42)],
                },
                Section {
                    constraints: Constraints {
                        addr: Some(Const::from_items(&[
                            LocationCounter.into(),
                            skipped_bytes.into(),
                            BinOp::Plus.into(),
                        ])),
                    },
                    addr: VarId(2),
                    size: VarId(3),
                    items: vec![Node::Byte(0x43)],
                },
            ],
            symbols: SymbolTable::new(),
            link_vars: 4,
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
        builder.finish();
        let relocs = program.resolve_relocs();
        assert_eq!(relocs[VarId(0)].value, addr.into());
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
            let name = object.symbols.alloc();
            let reloc = object.alloc_linkage_var();
            let items = &mut object.sections[0].items;
            items.push(Node::LdInlineAddr(0, Atom::Name(name.into()).into()));
            items.push(Node::Reloc(reloc));
            object
                .symbols
                .define(name, ProgramDef::Expr(Atom::Location(reloc).into()))
        })
    }

    #[test]
    fn resolve_expr_with_section_addr() {
        let program = Program {
            sections: vec![Section {
                constraints: Constraints {
                    addr: Some(0x1337.into()),
                },
                addr: VarId(0),
                size: VarId(1),
                items: vec![Node::Immediate(
                    Atom::Name(ProgramSymbol(0).into()).into(),
                    Width::Word,
                )],
            }],
            symbols: SymbolTable(vec![Some(ProgramDef::Section(SectionId(0)))]),
            link_vars: 2,
        };
        let binary = program.link(&mut IgnoreDiagnostics);
        assert_eq!(binary.sections[0].data, [0x37, 0x13])
    }

    #[test]
    fn traverse_reserved_bytes() {
        let addr = 0x0100;
        let bytes = 10;
        let symbol = VarId(2);
        let program = Program::<()> {
            sections: vec![Section {
                constraints: Constraints {
                    addr: Some(addr.into()),
                },
                addr: VarId(0),
                size: VarId(1),
                items: vec![
                    Node::Reserved(bytes.into()),
                    Node::Reloc(symbol),
                    Node::Immediate(Atom::Name(ProgramSymbol(0).into()).into(), Width::Word),
                ],
            }],
            symbols: SymbolTable(vec![Some(ProgramDef::Expr(Atom::Location(symbol).into()))]),
            link_vars: 3,
        };
        let relocs = program.resolve_relocs();
        assert_eq!(relocs[symbol].value, (addr + bytes).into())
    }

    fn assert_section_size(expected: impl Into<Num>, f: impl FnOnce(&mut Program<()>)) {
        let mut program = Program::new();
        program.add_section(None);
        f(&mut program);
        let relocs = program.resolve_relocs();
        assert_eq!(relocs[program.sections[0].size].value, expected.into())
    }
}
