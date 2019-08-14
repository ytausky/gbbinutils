use super::{BinarySection, LinkageContext, VarTable};

use crate::diag::{BackendDiagnostics, Message};
use crate::object::builder::Width;
use crate::object::{Const, Content, Node, Section};
use crate::span::Source;

use std::mem::replace;
use std::vec::IntoIter;

impl<S: Clone> Section<S> {
    pub(super) fn translate(
        &self,
        context: &mut LinkageContext<&Content<S>, &VarTable>,
        diagnostics: &mut impl BackendDiagnostics<S>,
    ) -> Vec<BinarySection> {
        let mut chunks = Vec::new();
        let mut data = Vec::new();
        let mut addr = context.vars[self.addr].value.clone();
        context.location = addr.clone();
        self.traverse(context, |item, context| {
            if let Node::Reserved(expr) = item {
                let bytes = expr.to_num(context, diagnostics);
                if !data.is_empty() {
                    chunks.push(BinarySection {
                        addr: addr.exact().unwrap() as usize,
                        data: replace(&mut data, Vec::new()),
                    });
                }
                context.location += &bytes;
                addr = context.location.clone();
            } else {
                data.extend(item.translate(context, diagnostics))
            }
        });
        if !data.is_empty() {
            chunks.push(BinarySection {
                addr: addr.exact().unwrap() as usize,
                data,
            });
        }
        chunks
    }
}

impl<S: Clone> Node<S> {
    fn translate(
        &self,
        context: &LinkageContext<&Content<S>, &VarTable>,
        diagnostics: &mut impl BackendDiagnostics<S>,
    ) -> IntoIter<u8> {
        match self {
            Node::Byte(value) => vec![*value],
            Node::Embedded(opcode, expr) => {
                let n = expr.to_num(context, diagnostics).exact().unwrap();
                vec![opcode | ((n as u8) << 3)]
            }
            Node::Immediate(expr, width) => {
                resolve_expr_item(&expr, *width, context, diagnostics).into_bytes()
            }
            Node::LdInlineAddr(opcode, expr) => {
                let addr = expr.to_num(context, diagnostics).exact().unwrap();
                let kind = if addr < 0xff00 {
                    AddrKind::Low
                } else {
                    AddrKind::High
                };
                let opcode = opcode
                    | match kind {
                        AddrKind::Low => 0x0a,
                        AddrKind::High => 0x00,
                    };
                let mut bytes = vec![opcode];
                let addr_repr = match kind {
                    AddrKind::Low => Data::Word(addr as u16),
                    AddrKind::High => Data::Byte((addr & 0xff) as u8),
                };
                bytes.extend(addr_repr.into_bytes());
                bytes
            }
            Node::Reloc(_) => vec![],
            Node::Reserved(_) => unimplemented!(),
        }
        .into_iter()
    }
}

#[derive(Clone, Copy)]
enum AddrKind {
    Low,
    High,
}

#[derive(Clone, Copy)]
enum Data {
    Byte(u8),
    Word(u16),
}

impl Data {
    fn into_bytes(self) -> Vec<u8> {
        match self {
            Data::Byte(value) => vec![value],
            Data::Word(value) => {
                let low = (value & 0xff) as u8;
                let high = ((value >> 8) & 0xff) as u8;
                vec![low, high]
            }
        }
    }
}

fn resolve_expr_item<S: Clone>(
    expr: &Const<S>,
    width: Width,
    context: &LinkageContext<&Content<S>, &VarTable>,
    diagnostics: &mut impl BackendDiagnostics<S>,
) -> Data {
    let span = expr.span();
    let value = expr.to_num(context, diagnostics).exact().unwrap_or(0);
    fit_to_width((value, span), width, diagnostics)
}

fn fit_to_width<S: Clone>(
    (value, value_ref): (i32, S),
    width: Width,
    diagnostics: &mut impl BackendDiagnostics<S>,
) -> Data {
    if !is_in_range(value, width) {
        diagnostics.emit_diag(Message::ValueOutOfRange { value, width }.at(value_ref))
    }
    match width {
        Width::Byte => Data::Byte(value as u8),
        Width::Word => Data::Word(value as u16),
    }
}

fn is_in_range(n: i32, width: Width) -> bool {
    match width {
        Width::Byte => is_in_byte_range(n),
        Width::Word => true,
    }
}

fn is_in_byte_range(n: i32) -> bool {
    is_in_i8_range(n) || is_in_u8_range(n)
}

fn is_in_i8_range(n: i32) -> bool {
    n >= i32::from(i8::min_value()) && n <= i32::from(i8::max_value())
}

fn is_in_u8_range(n: i32) -> bool {
    n >= i32::from(u8::min_value()) && n <= i32::from(u8::max_value())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::diag::IgnoreDiagnostics;
    use crate::model::{Atom, BinOp, LocationCounter};
    use crate::object::builder::*;
    use crate::object::num::Num;
    use crate::object::{Content, Object};

    use std::borrow::Borrow;

    #[test]
    fn translate_ld_deref_addr_a_with_low_addr() {
        test_translation_of_ld_inline_addr(0xe0, 0x2000, [0xea, 0x00, 0x20])
    }

    #[test]
    fn translate_ld_a_deref_addr_with_low_addr() {
        test_translation_of_ld_inline_addr(0xf0, 0x2000, [0xfa, 0x00, 0x20])
    }

    #[test]
    fn translate_ld_deref_addr_a_with_high_addr() {
        test_translation_of_ld_inline_addr(0xe0, 0xff77, [0xe0, 0x77])
    }

    #[test]
    fn translate_ld_a_deref_addr_with_high_addr() {
        test_translation_of_ld_inline_addr(0xf0, 0xff77, [0xf0, 0x77])
    }

    fn test_translation_of_ld_inline_addr(opcode: u8, addr: u16, expected: impl Borrow<[u8]>) {
        let actual =
            translate_section_item(Node::LdInlineAddr(opcode, Atom::Const(addr.into()).into()));
        assert_eq!(actual, expected.borrow())
    }

    #[test]
    fn translate_embedded() {
        let actual = translate_section_item(Node::Embedded(0b01_000_110, 4.into()));
        assert_eq!(actual, [0x66])
    }

    #[test]
    fn translate_expr_with_subtraction() {
        let actual = translate_section_item(Node::Immediate(
            Const::from_items(&[4.into(), 3.into(), BinOp::Minus.into()]),
            Width::Byte,
        ));
        assert_eq!(actual, [0x01])
    }

    fn translate_section_item<S: Clone + PartialEq>(item: Node<S>) -> Vec<u8> {
        item.translate(
            &LinkageContext {
                content: &Content::new(),
                vars: &VarTable(vec![]),
                location: Num::Unknown,
            },
            &mut IgnoreDiagnostics,
        )
        .collect()
    }

    #[test]
    fn set_addr_of_translated_section() {
        let addr = 0x7ff0;

        let mut object = Object::new();
        let object_builder = ObjectBuilder::new(&mut object);

        // org $7ff0
        let mut const_builder = object_builder.build_const();
        const_builder.push_op(addr, ());
        let (mut object_builder, origin) = const_builder.finish();
        object_builder.set_origin(origin);

        // nop
        object_builder.emit_item(Item::CpuInstr(CpuInstr::Nullary(Nullary::Nop)));

        object.vars.resolve(&object.content);
        let context = &mut LinkageContext {
            content: &object.content,
            vars: &object.vars,
            location: 0.into(),
        };
        let translated = object
            .content
            .sections()
            .next()
            .unwrap()
            .translate(context, &mut IgnoreDiagnostics);
        assert_eq!(translated[0].addr, addr as usize)
    }

    #[test]
    fn translate_expr_with_location_counter() {
        let mut object = Object::new();
        let mut object_builder = ObjectBuilder::new(&mut object);

        // nop
        object_builder.emit_item(Item::CpuInstr(CpuInstr::Nullary(Nullary::Nop)));

        // db .
        let mut const_builder = object_builder.build_const();
        const_builder.push_op(LocationCounter, ());
        let (mut object_builder, location) = const_builder.finish();
        object_builder.emit_item(Item::Data(location, Width::Byte));

        object.vars.resolve(&object.content);
        let context = &mut LinkageContext {
            content: &object.content,
            vars: &object.vars,
            location: 0.into(),
        };
        let binary = object
            .content
            .sections()
            .next()
            .unwrap()
            .translate(context, &mut IgnoreDiagnostics);
        assert_eq!(binary[0].data, [0x00, 0x02])
    }

    #[test]
    fn location_counter_starts_from_section_origin() {
        let addr = 0xffe1;

        let mut object = Object::new();
        let object_builder = ObjectBuilder::new(&mut object);

        // org $ffe1
        let mut const_builder = object_builder.build_const();
        const_builder.push_op(addr, ());
        let (mut object_builder, origin) = const_builder.finish();
        object_builder.set_origin(origin);

        // dw .
        let mut const_builder = object_builder.build_const();
        const_builder.push_op(LocationCounter, ());
        let (mut object_builder, location) = const_builder.finish();
        object_builder.emit_item(Item::Data(location, Width::Word));

        object.vars.resolve(&object.content);
        let context = &mut LinkageContext {
            content: &object.content,
            vars: &object.vars,
            location: 0.into(),
        };
        let binary = object
            .content
            .sections()
            .next()
            .unwrap()
            .translate(context, &mut IgnoreDiagnostics);
        assert_eq!(binary[0].data, [0xe3, 0xff])
    }
}
