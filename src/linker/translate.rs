use super::{LinkageContext, VarTable};

use crate::diagnostics::{BackendDiagnostics, Message};
use crate::object::{Content, Expr, Fragment, Section, Width};
use crate::program::Chunk;
use crate::span::Source;

use std::mem::replace;
use std::vec::IntoIter;

impl<'a, S: Clone + 'a> Section<S> {
    pub(super) fn translate<I>(
        &'a self,
        context: &mut LinkageContext<&'a Content<I, S>, &VarTable>,
        diagnostics: &mut impl BackendDiagnostics<S>,
    ) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut data = Vec::new();
        let mut addr = context.vars[self.addr].clone();
        context.location = addr.clone();
        self.traverse(context, |fragment, context| {
            if let Fragment::Reserved(expr) = fragment {
                let bytes = expr.to_num(context, diagnostics);
                if !data.is_empty() {
                    chunks.push(Chunk {
                        addr: addr.exact().unwrap() as usize,
                        data: replace(&mut data, Vec::new()).into_boxed_slice(),
                    });
                }
                context.location += &bytes;
                addr = context.location.clone();
            } else {
                data.extend(fragment.translate(context, diagnostics))
            }
        });
        if !data.is_empty() {
            chunks.push(Chunk {
                addr: addr.exact().unwrap() as usize,
                data: data.into_boxed_slice(),
            });
        }
        chunks
    }
}

impl<S: Clone> Fragment<Expr<S>> {
    fn translate<I>(
        &self,
        context: &LinkageContext<&Content<I, S>, &VarTable>,
        diagnostics: &mut impl BackendDiagnostics<S>,
    ) -> IntoIter<u8> {
        match self {
            Fragment::Byte(value) => vec![*value],
            Fragment::Embedded(opcode, expr) => {
                let n = expr.to_num(context, diagnostics).exact().unwrap();
                vec![opcode | ((n as u8) << 3)]
            }
            Fragment::Immediate(expr, width) => {
                resolve_expr_item(&expr, *width, context, diagnostics).into_bytes()
            }
            Fragment::LdInlineAddr(opcode, expr) => {
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
            Fragment::Reloc(_) => vec![],
            Fragment::Reserved(_) => unimplemented!(),
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

fn resolve_expr_item<I, S: Clone>(
    expr: &Expr<S>,
    width: Width,
    context: &LinkageContext<&Content<I, S>, &VarTable>,
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

    use crate::diagnostics::IgnoreDiagnostics;
    use crate::expr::{Atom, BinOp, Expr};
    use crate::object::var::Var;
    use crate::object::{Constraints, Content, Name, VarId};

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
        let actual = translate_section_item(Fragment::LdInlineAddr(
            opcode,
            Atom::Const(addr.into()).into(),
        ));
        assert_eq!(actual, expected.borrow())
    }

    #[test]
    fn translate_embedded() {
        let actual = translate_section_item(Fragment::Embedded(0b01_000_110, 4.into()));
        assert_eq!(actual, [0x66])
    }

    #[test]
    fn translate_expr_with_subtraction() {
        let actual = translate_section_item(Fragment::Immediate(
            Expr::from_items(&[4.into(), 3.into(), BinOp::Minus.into()]),
            Width::Byte,
        ));
        assert_eq!(actual, [0x01])
    }

    fn translate_section_item<S: Clone + PartialEq>(fragment: Fragment<Expr<Name, S>>) -> Vec<u8> {
        fragment
            .translate(
                &LinkageContext {
                    content: &Content::<&str, _>::new(),
                    vars: &VarTable::new(),
                    location: Var::Unknown,
                },
                &mut IgnoreDiagnostics,
            )
            .collect()
    }

    #[test]
    fn set_addr_of_translated_section() {
        let addr = 0x7ff0;

        // ORG $7ff0
        // NOP
        let content = Content::<&str, _> {
            sections: vec![Section {
                constraints: Constraints {
                    addr: Some(Expr::from_atom(Atom::Const(addr), ())),
                },
                addr: VarId(0),
                size: VarId(1),
                fragments: vec![Fragment::Byte(0x00)],
            }],
            symbols: vec![],
        };
        let mut vars = VarTable(vec![addr.into(), 1.into()]);

        vars.resolve(&content);
        let context = &mut LinkageContext {
            content: &content,
            vars: &vars,
            location: 0.into(),
        };
        let translated = content
            .sections()
            .next()
            .unwrap()
            .translate(context, &mut IgnoreDiagnostics);
        assert_eq!(translated[0].addr, addr as usize)
    }

    #[test]
    fn translate_expr_with_location_counter() {
        // NOP
        // DB   .
        let content = Content::<&str, _> {
            sections: vec![Section {
                constraints: Constraints { addr: None },
                addr: VarId(0),
                size: VarId(1),
                fragments: vec![
                    Fragment::Byte(0x00),
                    Fragment::Immediate(Expr::from_atom(Atom::Location, ()), Width::Byte),
                ],
            }],
            symbols: vec![],
        };
        let mut vars = VarTable(vec![0.into(), 2.into()]);

        vars.resolve(&content);
        let context = &mut LinkageContext {
            content: &content,
            vars: &vars,
            location: 0.into(),
        };
        let binary = content
            .sections()
            .next()
            .unwrap()
            .translate(context, &mut IgnoreDiagnostics);
        assert_eq!(*binary[0].data, [0x00, 0x02])
    }

    #[test]
    fn location_counter_starts_from_section_origin() {
        let addr = 0xffe1;

        // ORG  $ffe1
        // DW   .
        let content = Content::<&str, _> {
            sections: vec![Section {
                constraints: Constraints {
                    addr: Some(Expr::from_atom(Atom::Const(addr), ())),
                },
                addr: VarId(0),
                size: VarId(1),
                fragments: vec![Fragment::Immediate(
                    Expr::from_atom(Atom::Location, ()),
                    Width::Word,
                )],
            }],
            symbols: vec![],
        };
        let mut vars = VarTable(vec![addr.into(), 2.into()]);

        vars.resolve(&content);
        let context = &mut LinkageContext {
            content: &content,
            vars: &vars,
            location: 0.into(),
        };
        let binary = content
            .sections()
            .next()
            .unwrap()
            .translate(context, &mut IgnoreDiagnostics);
        assert_eq!(*binary[0].data, [0xe3, 0xff])
    }
}
