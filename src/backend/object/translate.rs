use super::context::{EvalContext, SymbolTable};
use super::{traverse_chunk_items, Chunk, Node};
use backend::{BinarySection, RelocExpr};
use diagnostics::{Diagnostic, DiagnosticsListener, Message, Source, SourceRange};
use std::vec::IntoIter;
use Width;

impl<SR: SourceRange> Chunk<SR> {
    pub fn translate(
        &self,
        context: &mut EvalContext<&SymbolTable>,
        diagnostics: &impl DiagnosticsListener<SR>,
    ) -> BinarySection {
        let mut data = Vec::<u8>::new();
        let origin = self.evaluate_origin(&context);
        context.location = origin.clone();
        traverse_chunk_items(&self.items, context, |item, context| {
            data.extend(item.translate(context, diagnostics))
        });
        BinarySection {
            origin: origin.exact().unwrap() as usize,
            data,
        }
    }
}

impl<SR: SourceRange> Node<SR> {
    fn translate(
        &self,
        context: &EvalContext<&SymbolTable>,
        diagnostics: &DiagnosticsListener<SR>,
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
            Node::Label(..) => vec![],
            Node::LdInlineAddr(opcode, expr) => {
                let addr = expr.evaluate(context).exact().unwrap();
                let kind = if addr < 0xff00 {
                    AddrKind::Low
                } else {
                    AddrKind::High
                };
                let opcode = opcode | match kind {
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
        }.into_iter()
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

fn resolve_expr_item<SR: SourceRange>(
    expr: &RelocExpr<SR>,
    width: Width,
    context: &EvalContext<&SymbolTable>,
    diagnostics: &DiagnosticsListener<SR>,
) -> Data {
    let range = expr.source_range();
    let value =
        expr.evaluate_strictly(context, &mut |symbol, range| {
            diagnostics.emit_diagnostic(Diagnostic::new(
                Message::UnresolvedSymbol {
                    symbol: symbol.to_string(),
                },
                range.clone(),
            ))
        }).exact()
            .unwrap_or(0);
    fit_to_width((value, range), width, diagnostics)
}

fn fit_to_width<SR: Clone>(
    (value, value_ref): (i32, SR),
    width: Width,
    diagnostics: &DiagnosticsListener<SR>,
) -> Data {
    if !is_in_range(value, width) {
        diagnostics.emit_diagnostic(Diagnostic::new(
            Message::ValueOutOfRange { value, width },
            value_ref,
        ))
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
    use backend::BinaryOperator;
    use diagnostics::IgnoreDiagnostics;
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
        let actual = translate_chunk_item(Node::LdInlineAddr(
            opcode,
            RelocExpr::Literal(addr.into(), ()),
        ));
        assert_eq!(actual, expected.borrow())
    }

    #[test]
    fn translate_embedded() {
        let actual = translate_chunk_item(Node::Embedded(0b01_000_110, RelocExpr::Literal(4, ())));
        assert_eq!(actual, [0x66])
    }

    #[test]
    fn translate_expr_with_subtraction() {
        let actual = translate_chunk_item(Node::Expr(
            RelocExpr::BinaryOperation(
                Box::new(RelocExpr::Literal(4, ())),
                Box::new(RelocExpr::Literal(3, ())),
                BinaryOperator::Minus,
                (),
            ),
            Width::Byte,
        ));
        assert_eq!(actual, [0x01])
    }

    fn translate_chunk_item<SR: SourceRange>(item: Node<SR>) -> Vec<u8> {
        use backend::object::resolve::Value;
        use diagnostics;
        item.translate(
            &EvalContext {
                symbols: &SymbolTable::new(),
                location: Value::Unknown,
            },
            &diagnostics::IgnoreDiagnostics {},
        ).collect()
    }

    #[test]
    fn set_origin_of_translated_chunk() {
        let addr = 0x7ff0;
        let chunk = Chunk {
            origin: Some(RelocExpr::Literal(addr, ())),
            items: Vec::new(),
        };
        let translated = translate_without_context(chunk);
        assert_eq!(translated.origin, addr as usize)
    }

    #[test]
    fn translate_expr_with_location_counter() {
        let byte = 0x42;
        let mut chunk = Chunk::new();
        chunk.items.extend(vec![
            Node::Byte(byte),
            Node::Expr(RelocExpr::LocationCounter(()), Width::Byte),
        ]);
        let binary = translate_without_context(chunk);
        assert_eq!(binary.data, [byte, 0x02])
    }

    #[test]
    fn location_counter_starts_from_chunk_origin() {
        let mut chunk = Chunk::new();
        chunk.origin = Some(RelocExpr::Literal(0xffe1, ()));
        chunk
            .items
            .push(Node::Expr(RelocExpr::LocationCounter(()), Width::Word));
        let binary = translate_without_context(chunk);
        assert_eq!(binary.data, [0xe3, 0xff])
    }

    fn translate_without_context<SR: SourceRange>(chunk: Chunk<SR>) -> BinarySection {
        let mut context = EvalContext {
            symbols: &SymbolTable::new(),
            location: 0.into(),
        };
        chunk.translate(&mut context, &IgnoreDiagnostics)
    }
}
