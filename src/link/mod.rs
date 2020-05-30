use crate::diag::{BackendDiagnostics, IgnoreDiagnostics};
use crate::object::num::Num;
use crate::object::*;
use crate::session::builder::Width;

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
                if let Node::Reloc(id) = item {
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
        F: FnMut(&Node<S>, &mut LinkageContext<&Content<S>, V>),
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
        context: &LinkageContext<&'a Content<S>, V>,
    ) -> Num {
        self.constraints
            .addr
            .as_ref()
            .map(|expr| expr.to_num(context, &mut IgnoreDiagnostics))
            .unwrap_or_else(|| 0.into())
    }
}

impl<S: Clone> Node<S> {
    fn size<'a, V: Borrow<VarTable>>(&self, context: &LinkageContext<&'a Content<S>, V>) -> Num {
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

    use crate::diag::{IgnoreDiagnostics, TestDiagnosticsListener};
    use crate::expr::*;
    use crate::session::builder::*;
    use crate::CompositeSession;

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
        let mut object = Object::new();
        let object_builder = CompositeSession {
            reentrancy: TestDiagnosticsListener::new(),
            names: (),
            builder: ObjectBuilder::new(&mut object),
        };

        // org $0150
        let mut const_builder = object_builder.build_const();
        const_builder.push_op(origin1, ());
        let (mut object_builder, origin1_const) = const_builder.finish();
        object_builder.set_origin(origin1_const.unwrap());

        // nop
        object_builder.emit_item(Item::CpuInstr(CpuInstr::Nullary(Nullary::Nop)));

        // org . + $10
        let mut const_builder = object_builder.build_const();
        const_builder.push_op(LocationCounter, ());
        const_builder.push_op(skipped_bytes, ());
        const_builder.push_op(BinOp::Plus, ());
        let (mut object_builder, origin2_const) = const_builder.finish();
        object_builder.set_origin(origin2_const.unwrap());

        // halt
        object_builder.emit_item(Item::CpuInstr(CpuInstr::Nullary(Nullary::Halt)));

        let binary = Program::link(object, &mut IgnoreDiagnostics);
        assert_eq!(
            binary.sections[1].addr,
            (origin1 + 1 + skipped_bytes) as usize
        )
    }

    #[test]
    fn label_defined_as_section_origin_plus_offset() {
        let addr = 0xffe1;
        let mut linkable = Object::new();
        let mut builder = CompositeSession {
            reentrancy: TestDiagnosticsListener::new(),
            names: (),
            builder: ObjectBuilder::new(&mut linkable),
        };
        builder.set_origin(addr.into());
        let symbol_id = builder.alloc_symbol(());
        let mut builder = builder.build_const();
        builder.push_op(LocationCounter, ());
        let (mut builder, expr) = builder.finish();
        builder.define_symbol(symbol_id, (), expr.unwrap());
        linkable.vars.resolve(&linkable.content);
        assert_eq!(linkable.vars[VarId(0)].value, addr.into());
    }

    #[test]
    fn empty_section_has_size_zero() {
        assert_section_size(0, |_| ())
    }

    #[test]
    fn section_with_one_byte_has_size_one() {
        assert_section_size(1, |mut builder| {
            builder.emit_item(Item::CpuInstr(CpuInstr::Nullary(Nullary::Nop)));
        });
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
        assert_section_size(expected, |mut builder| {
            builder.emit_item(Item::CpuInstr(CpuInstr::Ld(Ld::Special(
                SpecialLd::InlineAddr(addr.into()),
                Direction::IntoA,
            ))))
        });
    }

    #[test]
    fn ld_inline_addr_with_symbol_after_instruction_has_size_three() {
        assert_section_size(3, |mut builder| {
            let name = builder.alloc_symbol(());
            builder.emit_item(Item::CpuInstr(CpuInstr::Ld(Ld::Special(
                SpecialLd::InlineAddr(Atom::Name(name).into()),
                Direction::IntoA,
            ))));
            let mut symbol_builder = builder.build_const();
            symbol_builder.push_op(LocationCounter, ());
            let (mut builder, expr) = symbol_builder.finish();
            builder.define_symbol(name, (), expr.unwrap());
        })
    }

    #[test]
    fn resolve_expr_with_section_addr() {
        let mut object = Object::new();
        let mut object_builder = CompositeSession {
            reentrancy: TestDiagnosticsListener::new(),
            names: (),
            builder: ObjectBuilder::new(&mut object),
        };

        // section my_section
        let name = object_builder.alloc_symbol(());
        object_builder.start_section(name, ());

        // org $1337
        let mut const_builder = object_builder.build_const();
        const_builder.push_op(0x1337, ());
        let (mut object_builder, origin) = const_builder.finish();
        object_builder.set_origin(origin.unwrap());

        // dw my_section
        let mut const_builder = object_builder.build_const();
        const_builder.push_op(Name(name), ());
        let (mut object_builder, my_section) = const_builder.finish();
        object_builder.emit_item(Item::Data(my_section.unwrap(), Width::Word));

        let binary = Program::link(object, &mut IgnoreDiagnostics);
        assert_eq!(binary.sections[0].data, [0x37, 0x13])
    }

    #[test]
    fn traverse_reserved_bytes() {
        let addr = 0x0100;
        let bytes = 10;
        let symbol = VarId(2);

        let mut object = Object::new();
        let object_builder = CompositeSession {
            reentrancy: TestDiagnosticsListener::new(),
            names: (),
            builder: ObjectBuilder::new(&mut object),
        };

        // org $0100
        let mut const_builder = object_builder.build_const();
        const_builder.push_op(addr, ());
        let (mut object_builder, origin) = const_builder.finish();
        object_builder.set_origin(origin.unwrap());

        // ds 10
        let mut const_builder = object_builder.build_const();
        const_builder.push_op(bytes, ());
        let (mut object_builder, bytes_const) = const_builder.finish();
        object_builder.reserve(bytes_const.unwrap());

        // label dw label
        let label = object_builder.alloc_symbol(());
        let mut symbol_builder = object_builder.build_const();
        symbol_builder.push_op(LocationCounter, ());
        let (mut object_builder, expr) = symbol_builder.finish();
        object_builder.define_symbol(label, (), expr.unwrap());
        let mut const_builder = object_builder.build_const();
        const_builder.push_op(Name(label), ());
        let (mut object_builder, label_const) = const_builder.finish();
        object_builder.emit_item(Item::Data(label_const.unwrap(), Width::Word));

        object.vars.resolve(&object.content);
        assert_eq!(object.vars[symbol].value, (addr + bytes).into())
    }

    fn assert_section_size(expected: impl Into<Num>, f: impl FnOnce(Session<()>)) {
        let mut object = Object::new();
        let mut builder = CompositeSession {
            reentrancy: TestDiagnosticsListener::new(),
            names: (),
            builder: ObjectBuilder::new(&mut object),
        };
        let name = builder.alloc_symbol(());
        builder.start_section(name, ());
        f(builder);
        object.vars.resolve(&object.content);
        assert_eq!(
            object.vars[object.content.sections().next().unwrap().size].value,
            expected.into()
        );
    }

    type Session<'a, S> = CompositeSession<TestDiagnosticsListener<S>, (), ObjectBuilder<'a, S>>;
}
