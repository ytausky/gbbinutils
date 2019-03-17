use super::{EvalContext, RelocTable};

use crate::diag::{BackendDiagnostics, Message};
use crate::model::Width;
use crate::program::{BinarySection, Node, RelocExpr, Section};
use crate::span::Source;

use std::vec::IntoIter;

impl<S: Clone> Section<S> {
    pub(super) fn translate(
        &self,
        context: &mut EvalContext<&RelocTable>,
        diagnostics: &mut impl BackendDiagnostics<S>,
    ) -> BinarySection {
        let mut data = Vec::new();
        let (addr, _) = self.traverse(context, |item, context| {
            data.extend(item.translate(context, diagnostics))
        });
        BinarySection {
            name: self.name.clone().map(|name| name.into()),
            addr: addr.exact().unwrap() as usize,
            data,
        }
    }
}

impl<S: Clone> Node<S> {
    fn translate(
        &self,
        context: &EvalContext<&RelocTable>,
        diagnostics: &mut impl BackendDiagnostics<S>,
    ) -> IntoIter<u8> {
        match self {
            Node::Byte(value) => vec![*value],
            Node::Embedded(opcode, expr) => {
                let n = expr.evaluate(context).exact().unwrap();
                vec![opcode | ((n as u8) << 3)]
            }
            Node::Expr(expr, width) => {
                resolve_expr_item(&expr, *width, context, diagnostics).into_bytes()
            }
            Node::LdInlineAddr(opcode, expr) => {
                let addr = expr.evaluate(context).exact().unwrap();
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
            Node::Symbol(..) => vec![],
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
    expr: &RelocExpr<S>,
    width: Width,
    context: &EvalContext<&RelocTable>,
    diagnostics: &mut impl BackendDiagnostics<S>,
) -> Data {
    let span = expr.span();
    let value = expr
        .evaluate_strictly(context, &mut |span| {
            let symbol = diagnostics.strip_span(span);
            diagnostics.emit_diagnostic(Message::UnresolvedSymbol { symbol }.at(span.clone()))
        })
        .exact()
        .unwrap_or(0);
    fit_to_width((value, span), width, diagnostics)
}

fn fit_to_width<S: Clone>(
    (value, value_ref): (i32, S),
    width: Width,
    diagnostics: &mut impl BackendDiagnostics<S>,
) -> Data {
    if !is_in_range(value, width) {
        diagnostics.emit_diagnostic(Message::ValueOutOfRange { value, width }.at(value_ref))
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
    use crate::expr::{BinaryOperator, ExprVariant};
    use crate::model::Atom;
    use crate::program::{NameTable, RelocId};

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
        let actual = translate_section_item(Node::LdInlineAddr(
            opcode,
            Atom::Literal(addr.into()).into(),
        ));
        assert_eq!(actual, expected.borrow())
    }

    #[test]
    fn translate_embedded() {
        let actual = translate_section_item(Node::Embedded(0b01_000_110, 4.into()));
        assert_eq!(actual, [0x66])
    }

    #[test]
    fn translate_expr_with_subtraction() {
        let actual = translate_section_item(Node::Expr(
            ExprVariant::Binary(
                BinaryOperator::Minus,
                Box::new(4.into()),
                Box::new(3.into()),
            )
            .into(),
            Width::Byte,
        ));
        assert_eq!(actual, [0x01])
    }

    fn translate_section_item<S: Clone + PartialEq>(item: Node<S>) -> Vec<u8> {
        use crate::diag::IgnoreDiagnostics;
        use crate::program::link::Value;
        item.translate(
            &EvalContext {
                names: &NameTable::new(),
                relocs: &RelocTable::new(0),
                location: Value::Unknown,
            },
            &mut IgnoreDiagnostics::new(),
        )
        .collect()
    }

    #[test]
    fn set_addr_of_translated_section() {
        let addr = 0x7ff0;
        let section = Section {
            name: None,
            addr: Some(addr.into()),
            size: RelocId(0),
            items: Vec::new(),
        };
        let translated = translate_without_context(section);
        assert_eq!(translated.addr, addr as usize)
    }

    #[test]
    fn translate_expr_with_location_counter() {
        let byte = 0x42;
        let mut section = Section::new(None, RelocId(0));
        section.items.extend(vec![
            Node::Byte(byte),
            Node::Expr(Atom::LocationCounter.into(), Width::Byte),
        ]);
        let binary = translate_without_context(section);
        assert_eq!(binary.data, [byte, 0x02])
    }

    #[test]
    fn location_counter_starts_from_section_origin() {
        let mut section = Section::new(None, RelocId(0));
        section.addr = Some(0xffe1.into());
        section
            .items
            .push(Node::Expr(Atom::LocationCounter.into(), Width::Word));
        let binary = translate_without_context(section);
        assert_eq!(binary.data, [0xe3, 0xff])
    }

    #[test]
    fn translate_section_name() {
        let name = "my_section";
        let section = Section::<()>::new(Some(name.into()), RelocId(0));
        let binary = translate_without_context(section);
        assert_eq!(binary.name, Some(name.into()))
    }

    fn translate_without_context<S: Clone + PartialEq>(section: Section<S>) -> BinarySection {
        let mut context = EvalContext {
            names: &NameTable::new(),
            relocs: &RelocTable::new(0),
            location: 0.into(),
        };
        section.translate(&mut context, &mut IgnoreDiagnostics::new())
    }
}