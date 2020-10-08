use crate::diagnostics::{BackendDiagnostics, IgnoreDiagnostics};
use crate::object::num::Num;
use crate::object::*;

use std::borrow::Borrow;

mod translate;

pub struct Program {
    pub sections: Vec<BinarySection>,
}

impl Program {
    pub(crate) fn link<S: Clone>(
        mut object: Object<S>,
        diagnostics: &mut impl BackendDiagnostics<S>,
    ) -> Self {
        object.vars.resolve(&object.content);
        let mut context = LinkageContext {
            content: &object.content,
            vars: &object.vars,
            location: 0.into(),
        };
        Self {
            sections: object
                .content
                .sections()
                .flat_map(|section| section.translate(&mut context, diagnostics))
                .collect(),
        }
    }

    pub fn into_rom(self) -> Rom {
        let default = 0xffu8;
        let mut data: Vec<u8> = Vec::new();
        for section in self.sections {
            if !section.data.is_empty() {
                let end = section.addr + section.data.len();
                if data.len() < end {
                    data.resize(end, default)
                }
                data[section.addr..end].copy_from_slice(&section.data)
            }
        }
        if data.len() < MIN_ROM_LEN {
            data.resize(MIN_ROM_LEN, default)
        }
        Rom {
            data: data.into_boxed_slice(),
        }
    }
}

const MIN_ROM_LEN: usize = 0x8000;

pub struct Rom {
    pub data: Box<[u8]>,
}

pub struct BinarySection {
    pub addr: usize,
    pub data: Vec<u8>,
}

impl VarTable {
    fn resolve<S: Clone>(&mut self, content: &Content<S>) {
        self.refine_all(content);
        self.refine_all(content);
    }

    fn refine_all<S: Clone>(&mut self, content: &Content<S>) -> i32 {
        let mut refinements = 0;
        let context = &mut LinkageContext {
            content,
            vars: self,
            location: Num::Unknown,
        };
        for section in content.sections() {
            context.location = section.eval_addr(context);
            context.vars[section.addr].refine(context.location.clone());
            let size = section.traverse(context, |item, context| {
                if let Fragment::Reloc(id) = item {
                    refinements += context.vars[*id].refine(context.location.clone()) as i32
                }
            });
            refinements += context.vars[section.size].refine(size) as i32
        }
        refinements
    }
}

impl<S: Clone> Section<S> {
    fn traverse<V, F>(&self, context: &mut LinkageContext<&Content<S>, V>, mut f: F) -> Num
    where
        V: Borrow<VarTable>,
        F: FnMut(&Fragment<Expr<S>>, &mut LinkageContext<&Content<S>, V>),
    {
        let addr = context.location.clone();
        let mut offset = Num::from(0);
        for item in &self.fragments {
            offset += &item.size(&context);
            context.location = &addr + &offset;
            f(item, context)
        }
        offset
    }

    fn eval_addr<'a, V: Borrow<VarTable>>(
        &self,
        context: &LinkageContext<&'a Content<S>, V>,
    ) -> Num {
        self.constraints
            .addr
            .as_ref()
            .map(|expr| expr.to_num(context, &mut IgnoreDiagnostics))
            .unwrap_or_else(|| 0.into())
    }
}

impl<S: Clone> Fragment<Expr<S>> {
    fn size<'a, V: Borrow<VarTable>>(&self, context: &LinkageContext<&'a Content<S>, V>) -> Num {
        match self {
            Fragment::Byte(_) | Fragment::Embedded(..) => 1.into(),
            Fragment::Immediate(_, width) => width.len().into(),
            Fragment::LdInlineAddr(_, expr) => match expr.to_num(context, &mut IgnoreDiagnostics) {
                Num::Range { min, .. } if min >= 0xff00 => 2.into(),
                Num::Range { max, .. } if max < 0xff00 => 3.into(),
                _ => Num::Range { min: 2, max: 3 },
            },
            Fragment::Reloc(_) => 0.into(),
            Fragment::Reserved(bytes) => bytes.to_num(context, &mut IgnoreDiagnostics),
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

    use crate::diagnostics::IgnoreDiagnostics;
    use crate::expr::Expr;
    use crate::expr::*;
    use crate::span::WithSpan;

    #[test]
    fn empty_object_converted_to_all_0xff_rom() {
        let object = Program {
            sections: Vec::new(),
        };
        let rom = object.into_rom();
        assert_eq!(*rom.data, [0xffu8; MIN_ROM_LEN][..])
    }

    #[test]
    fn section_placed_in_rom_starting_at_origin() {
        let byte = 0x42;
        let addr = 0x150;
        let object = Program {
            sections: vec![BinarySection {
                addr,
                data: vec![byte],
            }],
        };
        let rom = object.into_rom();
        let mut expected = [0xffu8; MIN_ROM_LEN];
        expected[addr] = byte;
        assert_eq!(*rom.data, expected[..])
    }

    #[test]
    fn empty_section_does_not_extend_rom() {
        let addr = MIN_ROM_LEN + 1;
        let object = Program {
            sections: vec![BinarySection {
                addr,
                data: Vec::new(),
            }],
        };
        let rom = object.into_rom();
        assert_eq!(rom.data.len(), MIN_ROM_LEN)
    }

    #[test]
    fn resolve_origin_relative_to_previous_section() {
        let origin1 = 0x150;
        let skipped_bytes = 0x10;

        // ORG $0150
        // NOP
        // ORG . + $10
        // HALT
        let object = Object {
            content: Content {
                sections: vec![
                    Section {
                        constraints: Constraints {
                            addr: Some(Expr::from_atom(Atom::Const(0x0150), ())),
                        },
                        addr: VarId(0),
                        size: VarId(1),
                        fragments: vec![Fragment::Byte(0x00)],
                    },
                    Section {
                        constraints: Constraints {
                            addr: Some(Expr(vec![
                                ExprOp::Atom(Atom::Location).with_span(()),
                                ExprOp::Atom(Atom::Const(0x10)).with_span(()),
                                ExprOp::Binary(BinOp::Plus).with_span(()),
                            ])),
                        },
                        addr: VarId(2),
                        size: VarId(3),
                        fragments: vec![Fragment::Byte(0x76)],
                    },
                ],
                symbols: SymbolTable(vec![]),
            },
            vars: VarTable(vec![
                Var {
                    value: 0x0150.into(),
                },
                Var { value: 1.into() },
                Var {
                    value: (0x0150 + 1 + 0x10).into(),
                },
                Var { value: 1.into() },
            ]),
        };

        let binary = Program::link(object, &mut IgnoreDiagnostics);
        assert_eq!(
            binary.sections[1].addr,
            (origin1 + 1 + skipped_bytes) as usize
        )
    }

    #[test]
    fn label_defined_as_section_origin_plus_offset() {
        let addr = 0xffe1;

        // ORG $ffe1
        // LABEL
        let mut object = Object {
            content: Content {
                sections: vec![Section {
                    constraints: Constraints {
                        addr: Some(Expr::from_atom(Atom::Const(addr), ())),
                    },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![Fragment::Reloc(VarId(2))],
                }],
                symbols: SymbolTable(vec![Some(UserDef::Closure(Closure {
                    expr: Expr::from_atom(Atom::Location, ()),
                    location: VarId(2),
                }))]),
            },
            vars: VarTable(vec![
                Var { value: addr.into() },
                Var { value: 0.into() },
                Var { value: addr.into() },
            ]),
        };

        object.vars.resolve(&object.content);
        assert_eq!(object.vars[VarId(0)].value, addr.into());
    }

    #[test]
    fn empty_section_has_size_zero() {
        assert_section_size(
            0,
            Object {
                content: Content {
                    sections: vec![Section {
                        constraints: Constraints { addr: None },
                        addr: VarId(0),
                        size: VarId(1),
                        fragments: vec![],
                    }],
                    symbols: SymbolTable::new(),
                },
                vars: VarTable(vec![
                    Var {
                        value: 0x0000.into(),
                    },
                    Var { value: 0.into() },
                ]),
            },
        )
    }

    #[test]
    fn section_with_one_byte_has_size_one() {
        assert_section_size(
            1,
            Object {
                content: Content {
                    sections: vec![Section {
                        constraints: Constraints { addr: None },
                        addr: VarId(0),
                        size: VarId(1),
                        fragments: vec![Fragment::Byte(0x00)],
                    }],
                    symbols: SymbolTable::new(),
                },
                vars: VarTable(vec![
                    Var {
                        value: 0x0000.into(),
                    },
                    Var { value: 1.into() },
                ]),
            },
        );
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
        assert_section_size(
            expected,
            Object {
                content: Content {
                    sections: vec![Section {
                        constraints: Constraints { addr: None },
                        addr: VarId(0),
                        size: VarId(1),
                        fragments: vec![Fragment::LdInlineAddr(0xf0, addr.into())],
                    }],
                    symbols: SymbolTable::new(),
                },
                vars: VarTable(vec![
                    Var {
                        value: 0x0000.into(),
                    },
                    Var {
                        value: Num::Unknown,
                    },
                ]),
            },
        );
    }

    #[test]
    fn ld_inline_addr_with_symbol_after_instruction_has_size_three() {
        assert_section_size(
            3,
            Object {
                content: Content {
                    sections: vec![Section {
                        constraints: Constraints { addr: None },
                        addr: VarId(0),
                        size: VarId(1),
                        fragments: vec![
                            Fragment::LdInlineAddr(
                                0xf0,
                                Atom::Name(Symbol::UserDef(UserDefId(0))).into(),
                            ),
                            Fragment::Reloc(VarId(2)),
                        ],
                    }],
                    symbols: SymbolTable(vec![Some(UserDef::Closure(Closure {
                        expr: Expr::from_atom(Atom::Location, ()),
                        location: VarId(2),
                    }))]),
                },
                vars: VarTable(vec![
                    Var {
                        value: 0x0000.into(),
                    },
                    Var {
                        value: Num::Range { min: 2, max: 3 },
                    },
                    Var {
                        value: Num::Range { min: 2, max: 3 },
                    },
                ]),
            },
        )
    }

    #[test]
    fn resolve_expr_with_section_addr() {
        // my_section   SECTION
        //              ORG     $1337
        //              DW      my_section
        let object = Object {
            content: Content {
                sections: vec![Section {
                    constraints: Constraints {
                        addr: Some(Expr::from_atom(Atom::Const(0x1337), ())),
                    },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![Fragment::Immediate(
                        Expr::from_atom(Atom::Name(Symbol::UserDef(UserDefId(0))), ()),
                        Width::Word,
                    )],
                }],
                symbols: SymbolTable(vec![Some(UserDef::Section(SectionId(0)))]),
            },
            vars: VarTable(vec![
                Var {
                    value: 0x1337.into(),
                },
                Var { value: 2.into() },
            ]),
        };

        let binary = Program::link(object, &mut IgnoreDiagnostics);
        assert_eq!(binary.sections[0].data, [0x37, 0x13])
    }

    #[test]
    fn traverse_reserved_bytes() {
        let addr = 0x0100;
        let bytes = 10;
        let symbol = VarId(2);

        //          ORG $0100
        //          DS  10
        // label    DW  label
        let mut object = Object {
            content: Content {
                sections: vec![Section {
                    constraints: Constraints {
                        addr: Some(Expr::from_atom(Atom::Const(addr), ())),
                    },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![
                        Fragment::Reserved(Expr::from_atom(Atom::Const(bytes), ())),
                        Fragment::Reloc(VarId(2)),
                        Fragment::Immediate(
                            Expr::from_atom(Atom::Name(Symbol::UserDef(UserDefId(0))), ()),
                            Width::Word,
                        ),
                    ],
                }],
                symbols: SymbolTable(vec![Some(UserDef::Closure(Closure {
                    expr: Expr::from_atom(Atom::Name(Symbol::UserDef(UserDefId(0))), ()),
                    location: VarId(2),
                }))]),
            },
            vars: VarTable(vec![
                Var { value: addr.into() },
                Var {
                    value: (bytes + 2).into(),
                },
                Var {
                    value: (addr + bytes).into(),
                },
            ]),
        };

        object.vars.resolve(&object.content);
        assert_eq!(object.vars[symbol].value, (addr + bytes).into())
    }

    fn assert_section_size(expected: impl Into<Num>, mut object: Object<()>) {
        object.vars.resolve(&object.content);
        assert_eq!(
            object.vars[object.content.sections().next().unwrap().size].value,
            expected.into()
        );
    }
}
