pub use self::context::LinkingContext;

use self::context::ChunkSize;
use backend::{BinaryObject, BinarySection, Chunk, Node, Object};
use diagnostics::{Diagnostic, DiagnosticsListener, Message, Source, SourceInterval};
use instruction::{Direction, RelocExpr};
use std::ops::AddAssign;
use std::vec::IntoIter;
use Width;

mod context;

#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    min: i32,
    max: i32,
}

impl Value {
    fn exact(&self) -> Option<i32> {
        if self.min == self.max {
            Some(self.min)
        } else {
            None
        }
    }

    fn len(&self) -> i32 {
        self.max - self.min
    }
}

impl From<i32> for Value {
    fn from(n: i32) -> Self {
        Value { min: n, max: n }
    }
}

impl AddAssign<Value> for Value {
    fn add_assign(&mut self, rhs: Value) {
        self.min += rhs.min;
        self.max += rhs.max
    }
}

pub fn link<'a, SR, D>(object: Object<SR>, diagnostics: &D) -> BinaryObject
where
    SR: SourceInterval,
    D: DiagnosticsListener<SR> + 'a,
{
    let symbols = resolve_symbols(&object);
    BinaryObject {
        sections: object
            .chunks
            .into_iter()
            .map(|section| resolve_section(section, &symbols, diagnostics))
            .collect(),
    }
}

fn resolve_section<SR: SourceInterval>(
    section: Chunk<SR>,
    context: &LinkingContext,
    diagnostics: &impl DiagnosticsListener<SR>,
) -> BinarySection {
    BinarySection {
        data: section
            .items
            .into_iter()
            .flat_map(|node| match node.translate(context) {
                Ok(iter) => iter,
                Err(diagnostic) => {
                    diagnostics.emit_diagnostic(diagnostic);
                    Vec::new().into_iter()
                }
            })
            .collect(),
    }
}

fn resolve_symbols<SR: Clone>(object: &Object<SR>) -> LinkingContext {
    let mut symbols = collect_symbols(object);
    refine_symbols(object, &mut symbols);
    symbols
}

fn collect_symbols<SR: Clone>(object: &Object<SR>) -> LinkingContext {
    let mut symbols = LinkingContext::new();
    (0..object.chunks.len()).for_each(|i| symbols.define(ChunkSize(i), None));
    for (i, chunk) in (&object.chunks).into_iter().enumerate() {
        let mut location = Value::from(0);
        for node in &chunk.items {
            match node {
                Node::Label(symbol, _) => symbols.define(symbol.as_str(), Some(location.clone())),
                node => location += node.size(&symbols),
            }
        }
        symbols.refine(ChunkSize(i), location);
    }
    symbols
}

fn refine_symbols<SR: Clone>(object: &Object<SR>, context: &mut LinkingContext) -> i32 {
    let mut refinements = 0;
    for (i, chunk) in (&object.chunks).into_iter().enumerate() {
        let mut location = Value::from(0);
        for node in &chunk.items {
            match node {
                Node::Label(symbol, _) => {
                    refinements += context.refine(symbol.as_str(), location.clone()) as i32
                }
                node => location += node.size(context),
            }
        }
        refinements += context.refine(ChunkSize(i), location) as i32
    }
    refinements
}

#[derive(Debug)]
struct UndefinedSymbol<SR>(String, SR);

impl<SR: Clone> RelocExpr<SR> {
    fn evaluate(&self, context: &LinkingContext) -> Result<Option<Value>, UndefinedSymbol<SR>> {
        match self {
            RelocExpr::Literal(value, _) => Ok(Some((*value).into())),
            RelocExpr::LocationCounter => panic!(),
            RelocExpr::Subtract(_, _) => panic!(),
            RelocExpr::Symbol(symbol, expr_ref) => context
                .get(symbol.as_str())
                .cloned()
                .ok_or_else(|| UndefinedSymbol((*symbol).clone(), (*expr_ref).clone())),
        }
    }
}

fn resolve_expr_item<SR: SourceInterval>(
    expr: &RelocExpr<SR>,
    width: Width,
    context: &LinkingContext,
) -> Result<Data, Diagnostic<SR>> {
    let range = expr.source_interval();
    let value = expr.evaluate(context)
        .map_err(|undefined| {
            let UndefinedSymbol(symbol, range) = undefined;
            Diagnostic::new(Message::UnresolvedSymbol { symbol }, range)
        })?
        .unwrap()
        .exact()
        .unwrap();
    fit_to_width((value, range), width)
}

fn fit_to_width<SR: Clone>(
    (value, value_ref): (i32, SR),
    width: Width,
) -> Result<Data, Diagnostic<SR>> {
    if !is_in_range(value, width) {
        Err(Diagnostic::new(
            Message::ValueOutOfRange { value, width },
            value_ref.clone(),
        ))
    } else {
        Ok(match width {
            Width::Byte => Data::Byte(value as u8),
            Width::Word => Data::Word(value as u16),
        })
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

impl<SR: Clone> Node<SR> {
    fn size(&self, symbols: &LinkingContext) -> Value {
        match self {
            Node::Byte(_) | Node::Embedded(..) => 1.into(),
            Node::Expr(_, width) => width.len().into(),
            Node::Label(..) => 0.into(),
            Node::LdInlineAddr(expr, _) => match expr.evaluate(symbols) {
                Ok(Some(Value { min, .. })) if min >= 0xff00 => 2.into(),
                Ok(Some(Value { max, .. })) if max < 0xff00 => 3.into(),
                _ => Value { min: 2, max: 3 },
            },
        }
    }
}

impl<SR: SourceInterval> Node<SR> {
    fn translate(&self, context: &LinkingContext) -> Result<IntoIter<u8>, Diagnostic<SR>> {
        Ok(match self {
            Node::Byte(value) => vec![*value],
            Node::Embedded(..) => panic!(),
            Node::Expr(expr, width) => {
                resolve_expr_item(&expr, *width, context).map(|data| data.into_bytes())?
            }
            Node::Label(..) => vec![],
            Node::LdInlineAddr(expr, direction) => {
                let addr = expr.evaluate(context).unwrap().unwrap().exact().unwrap();
                let kind = if addr < 0xff00 {
                    AddrKind::Low
                } else {
                    AddrKind::High
                };
                let opcode: u8 = match (kind, direction) {
                    (AddrKind::Low, Direction::FromA) => 0xea,
                    (AddrKind::Low, Direction::IntoA) => 0xfa,
                    (AddrKind::High, Direction::FromA) => 0xe0,
                    (AddrKind::High, Direction::IntoA) => 0xf0,
                };
                let mut bytes = vec![opcode];
                let addr_repr = match kind {
                    AddrKind::Low => Data::Word(addr as u16),
                    AddrKind::High => Data::Byte((addr & 0xff) as u8),
                };
                bytes.extend(addr_repr.into_bytes());
                bytes
            }
        }.into_iter())
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

#[cfg(test)]
mod tests {
    use super::*;
    use instruction::Direction;
    use std::borrow::Borrow;

    #[test]
    fn translate_ld_deref_addr_a_with_low_addr() {
        test_translation_of_ld_inline_addr(0x2000, Direction::FromA, [0xea, 0x00, 0x20])
    }

    #[test]
    fn translate_ld_a_deref_addr_with_low_addr() {
        test_translation_of_ld_inline_addr(0x2000, Direction::IntoA, [0xfa, 0x00, 0x20])
    }

    #[test]
    fn translate_ld_deref_addr_a_with_high_addr() {
        test_translation_of_ld_inline_addr(0xff77, Direction::FromA, [0xe0, 0x77])
    }

    #[test]
    fn translate_ld_a_deref_addr_with_high_addr() {
        test_translation_of_ld_inline_addr(0xff77, Direction::IntoA, [0xf0, 0x77])
    }

    fn test_translation_of_ld_inline_addr(
        addr: u16,
        direction: Direction,
        expected: impl Borrow<[u8]>,
    ) {
        let actual: Vec<_> = Node::LdInlineAddr(RelocExpr::Literal(addr.into(), ()), direction)
            .translate(&LinkingContext::new())
            .unwrap()
            .collect();
        assert_eq!(actual, expected.borrow())
    }

    #[test]
    fn empty_chunk_has_size_zero() {
        assert_chunk_size(0, |_| ())
    }

    #[test]
    fn chunk_with_one_byte_has_size_one() {
        assert_chunk_size(1, |section| section.items.push(Node::Byte(0x42)));
    }

    #[test]
    fn chunk_with_const_inline_addr_ld_has_size_two() {
        test_chunk_size_with_literal_ld_inline_addr(0xff00, 2)
    }

    #[test]
    fn chunk_with_const_inline_addr_ld_has_size_three() {
        test_chunk_size_with_literal_ld_inline_addr(0x1234, 3)
    }

    fn test_chunk_size_with_literal_ld_inline_addr(addr: i32, expected: i32) {
        assert_chunk_size(expected, |section| {
            section.items.push(Node::LdInlineAddr(
                RelocExpr::Literal(addr, ()),
                Direction::FromA,
            ))
        });
    }

    #[test]
    fn ld_inline_addr_with_symbol_after_instruction_has_size_three() {
        assert_chunk_size(3, |section| {
            section.items.extend(
                [
                    Node::LdInlineAddr(
                        RelocExpr::Symbol("label".to_string(), ()),
                        Direction::FromA,
                    ),
                    Node::Label("label".to_string(), ()),
                ].iter()
                    .cloned(),
            )
        })
    }

    fn assert_chunk_size(expected: impl Into<Value>, f: impl FnOnce(&mut Chunk<()>)) {
        let mut object = Object::<()>::new();
        object.add_chunk();
        f(&mut object.chunks[0]);
        let symbols = resolve_symbols(&object);
        assert_eq!(
            symbols.get(ChunkSize(0)).cloned(),
            Some(Some(expected.into()))
        )
    }
}
