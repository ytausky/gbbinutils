pub use self::context::{EvalContext, SymbolTable};

use self::context::ChunkSize;
use backend::{BinaryObject, BinarySection, Chunk, Node, Object, RelocExpr};
use diagnostics::{Diagnostic, DiagnosticsListener, Message, Source, SourceInterval};
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
    symbols: &SymbolTable,
    diagnostics: &impl DiagnosticsListener<SR>,
) -> BinarySection {
    let mut data = Vec::<u8>::new();
    let mut context = EvalContext {
        symbols,
        location: Some(0.into()),
    };
    for item in section.items {
        let size = item.size(&context);
        match item.translate(&context) {
            Ok(iter) => data.extend(iter),
            Err(diagnostic) => diagnostics.emit_diagnostic(diagnostic),
        }
        *context.location.as_mut().unwrap() += size;
    }
    BinarySection { data }
}

fn resolve_symbols<SR: Clone>(object: &Object<SR>) -> SymbolTable {
    let mut symbols = collect_symbols(object);
    refine_symbols(object, &mut symbols);
    symbols
}

fn collect_symbols<SR: Clone>(object: &Object<SR>) -> SymbolTable {
    let mut symbols = SymbolTable::new();
    (0..object.chunks.len()).for_each(|i| symbols.define(ChunkSize(i), None));
    for (i, chunk) in (&object.chunks).into_iter().enumerate() {
        let size = chunk.traverse(&mut symbols, |location, item, context| {
            if let Node::Label(symbol, _) = item {
                context.define(symbol.as_str(), Some(location.clone()))
            }
        });
        symbols.refine(ChunkSize(i), size);
    }
    symbols
}

fn refine_symbols<SR: Clone>(object: &Object<SR>, context: &mut SymbolTable) -> i32 {
    let mut refinements = 0;
    for (i, chunk) in (&object.chunks).into_iter().enumerate() {
        let size = chunk.traverse(context, |location, item, context| {
            if let Node::Label(symbol, _) = item {
                refinements += context.refine(symbol.as_str(), location.clone()) as i32
            }
        });
        refinements += context.refine(ChunkSize(i), size) as i32
    }
    refinements
}

impl<SR: Clone> Chunk<SR> {
    fn traverse<F>(&self, symbols: &mut SymbolTable, mut f: F) -> Value
    where
        F: FnMut(&Value, &Node<SR>, &mut SymbolTable),
    {
        let mut location = Value::from(0);
        for item in &self.items {
            f(&location, item, symbols);
            location += {
                let context = EvalContext {
                    symbols,
                    location: Some(location.clone()),
                };
                item.size(&context)
            }
        }
        location
    }
}

#[derive(Debug)]
struct UndefinedSymbol<SR>(String, SR);

impl<SR: Clone> RelocExpr<SR> {
    fn evaluate(&self, context: &EvalContext) -> Result<Option<Value>, UndefinedSymbol<SR>> {
        match self {
            RelocExpr::Literal(value, _) => Ok(Some((*value).into())),
            RelocExpr::LocationCounter(_) => Ok(context.location.clone()),
            RelocExpr::Subtract(_, _) => panic!(),
            RelocExpr::Symbol(symbol, expr_ref) => context
                .symbols
                .get(symbol.as_str())
                .cloned()
                .ok_or_else(|| UndefinedSymbol((*symbol).clone(), (*expr_ref).clone())),
        }
    }
}

fn resolve_expr_item<SR: SourceInterval>(
    expr: &RelocExpr<SR>,
    width: Width,
    context: &EvalContext,
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
    fn size(&self, context: &EvalContext) -> Value {
        match self {
            Node::Byte(_) | Node::Embedded(..) => 1.into(),
            Node::Expr(_, width) => width.len().into(),
            Node::Label(..) => 0.into(),
            Node::LdInlineAddr(_, expr) => match expr.evaluate(context) {
                Ok(Some(Value { min, .. })) if min >= 0xff00 => 2.into(),
                Ok(Some(Value { max, .. })) if max < 0xff00 => 3.into(),
                _ => Value { min: 2, max: 3 },
            },
        }
    }
}

impl<SR: SourceInterval> Node<SR> {
    fn translate(&self, context: &EvalContext) -> Result<IntoIter<u8>, Diagnostic<SR>> {
        Ok(match self {
            Node::Byte(value) => vec![*value],
            Node::Embedded(opcode, expr) => {
                let n = expr.evaluate(context).unwrap().unwrap().exact().unwrap();
                vec![opcode | ((n as u8) << 3)]
            }
            Node::Expr(expr, width) => {
                resolve_expr_item(&expr, *width, context).map(|data| data.into_bytes())?
            }
            Node::Label(..) => vec![],
            Node::LdInlineAddr(opcode, expr) => {
                let addr = expr.evaluate(context).unwrap().unwrap().exact().unwrap();
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
    use diagnostics::TestDiagnosticsListener;
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
        let actual: Vec<_> = Node::LdInlineAddr(opcode, RelocExpr::Literal(addr.into(), ()))
            .translate(&EvalContext {
                symbols: &SymbolTable::new(),
                location: None,
            })
            .unwrap()
            .collect();
        assert_eq!(actual, expected.borrow())
    }

    #[test]
    fn translate_embedded() {
        let actual: Vec<_> = Node::Embedded(0b01_000_110, RelocExpr::Literal(4, ()))
            .translate(&EvalContext {
                symbols: &SymbolTable::new(),
                location: None,
            })
            .unwrap()
            .collect();
        assert_eq!(actual, [0x66])
    }

    #[test]
    fn translate_expr_with_location_counter() {
        let byte = 0x42;
        let mut chunk = Chunk::new();
        chunk.items.extend(vec![
            Node::Byte(byte),
            Node::Expr(RelocExpr::LocationCounter(()), Width::Byte),
        ]);
        let binary = resolve_section(chunk, &SymbolTable::new(), &TestDiagnosticsListener::new());
        assert_eq!(binary.data, [byte, 0x01])
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
            section
                .items
                .push(Node::LdInlineAddr(0, RelocExpr::Literal(addr, ())))
        });
    }

    #[test]
    fn ld_inline_addr_with_symbol_after_instruction_has_size_three() {
        assert_chunk_size(3, |section| {
            section.items.extend(
                [
                    Node::LdInlineAddr(0, RelocExpr::Symbol("label".to_string(), ())),
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
