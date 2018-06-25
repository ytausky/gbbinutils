pub use self::context::LinkingContext;

use self::context::ChunkSize;
use backend::{BinaryObject, BinarySection, Chunk, Node, Object, Value};
use diagnostics::{DiagnosticsListener, SourceInterval};
use instruction::RelocExpr;

mod context;

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
    section
        .items
        .into_iter()
        .flat_map(|node| node.translate(context, diagnostics))
        .collect()
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

impl<SR: Clone> Node<SR> {
    pub fn size(&self, symbols: &LinkingContext) -> Value {
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

#[cfg(test)]
mod tests {
    use super::*;
    use instruction::Direction;

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
