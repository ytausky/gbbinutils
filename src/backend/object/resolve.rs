pub use super::context::{EvalContext, SymbolTable};

use super::context::ChunkSize;
use crate::backend::{Node, Object, RelocAtom, RelocExpr};
use crate::expr::ExprVariant;
use crate::span::{Source, Span};
use std::borrow::Borrow;
use std::ops::{Add, AddAssign, Sub};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Range { min: i32, max: i32 },
    Unknown,
}

impl Value {
    pub fn exact(&self) -> Option<i32> {
        match *self {
            Value::Range { min, max } if min == max => Some(min),
            _ => None,
        }
    }
}

impl From<i32> for Value {
    fn from(n: i32) -> Self {
        Value::Range { min: n, max: n }
    }
}

impl AddAssign<Value> for Value {
    fn add_assign(&mut self, rhs: Value) {
        match (self, rhs) {
            (
                Value::Range { min, max },
                Value::Range {
                    min: rhs_min,
                    max: rhs_max,
                },
            ) => {
                *min += rhs_min;
                *max += rhs_max;
            }
            (this, _) => *this = Value::Unknown,
        }
    }
}

impl<T: Into<Value>> Add<T> for Value {
    type Output = Value;
    fn add(mut self, rhs: T) -> Self::Output {
        self += rhs.into();
        self
    }
}

impl Sub<Value> for Value {
    type Output = Value;
    fn sub(self, rhs: Value) -> Self::Output {
        match (self, rhs) {
            (
                Value::Range { min, max },
                Value::Range {
                    min: rhs_min,
                    max: rhs_max,
                },
            ) => Value::Range {
                min: min - rhs_max,
                max: max - rhs_min,
            },
            _ => Value::Unknown,
        }
    }
}

pub fn resolve_symbols<S: Span>(object: &Object<S>) -> SymbolTable {
    let mut symbols = collect_symbols(object);
    refine_symbols(object, &mut symbols);
    symbols
}

fn collect_symbols<S: Span>(object: &Object<S>) -> SymbolTable {
    let mut symbols = SymbolTable::new();
    (0..object.chunks.len()).for_each(|i| symbols.define(ChunkSize(i), Value::Unknown));
    {
        let mut context = EvalContext {
            symbols: &mut symbols,
            location: Value::Unknown,
        };
        for (i, chunk) in object.chunks.iter().enumerate() {
            let size = chunk.traverse(&mut context, |item, context| {
                if let Node::Label(symbol, _) = item {
                    context
                        .symbols
                        .define(symbol.as_str(), context.location.clone())
                }
            });
            context.symbols.refine(ChunkSize(i), size);
        }
    }
    symbols
}

fn refine_symbols<S: Span>(object: &Object<S>, symbols: &mut SymbolTable) -> i32 {
    let mut refinements = 0;
    let context = &mut EvalContext {
        symbols,
        location: Value::Unknown,
    };
    for (i, chunk) in object.chunks.iter().enumerate() {
        let size = chunk.traverse(context, |item, context| {
            if let Node::Label(symbol, _) = item {
                refinements += context
                    .symbols
                    .refine(symbol.as_str(), context.location.clone())
                    as i32
            }
        });
        refinements += context.symbols.refine(ChunkSize(i), size) as i32
    }
    refinements
}

impl<S: Span> RelocExpr<S> {
    pub fn evaluate<ST: Borrow<SymbolTable>>(&self, context: &EvalContext<ST>) -> Value {
        self.evaluate_strictly(context, &mut |_: &str, _: &S| ())
    }

    pub fn evaluate_strictly<ST, F>(
        &self,
        context: &EvalContext<ST>,
        on_undefined_symbol: &mut F,
    ) -> Value
    where
        ST: Borrow<SymbolTable>,
        F: FnMut(&str, &S),
    {
        match &self.variant {
            ExprVariant::Unary(_, _) => unreachable!(),
            ExprVariant::Binary(operator, lhs, rhs) => {
                use crate::backend::BinaryOperator;
                let lhs = lhs.evaluate_strictly(context, on_undefined_symbol);
                let rhs = rhs.evaluate_strictly(context, on_undefined_symbol);
                match operator {
                    BinaryOperator::Minus => lhs - rhs,
                    BinaryOperator::Plus => lhs + rhs,
                }
            }
            ExprVariant::Atom(RelocAtom::Literal(value)) => (*value).into(),
            ExprVariant::Atom(RelocAtom::LocationCounter) => context.location.clone(),
            ExprVariant::Atom(RelocAtom::Symbol(symbol)) => context
                .symbols
                .borrow()
                .get(symbol.as_str())
                .cloned()
                .unwrap_or_else(|| {
                    on_undefined_symbol(&symbol, &self.span());
                    Value::Unknown
                }),
        }
    }
}

impl<S: Span> Node<S> {
    pub fn size<ST: Borrow<SymbolTable>>(&self, context: &EvalContext<ST>) -> Value {
        match self {
            Node::Byte(_) | Node::Embedded(..) => 1.into(),
            Node::Expr(_, width) => width.len().into(),
            Node::Label(..) => 0.into(),
            Node::LdInlineAddr(_, expr) => match expr.evaluate(context) {
                Value::Range { min, .. } if min >= 0xff00 => 2.into(),
                Value::Range { max, .. } if max < 0xff00 => 3.into(),
                _ => Value::Range { min: 2, max: 3 },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::object::Chunk;
    use crate::backend::{object::ObjectBuilder, Backend};

    #[test]
    fn label_defined_as_chunk_origin_plus_offset() {
        let label = "label";
        let addr = 0xffe1;
        let mut builder = ObjectBuilder::new();
        builder.set_origin(addr.into());
        builder.add_label((label, ()));
        let object = builder.into_object();
        let symbols = resolve_symbols(&object);
        assert_eq!(symbols.get(label), Some(&addr.into()))
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
            section.items.push(Node::LdInlineAddr(0, addr.into()))
        });
    }

    #[test]
    fn ld_inline_addr_with_symbol_after_instruction_has_size_three() {
        assert_chunk_size(3, |section| {
            section.items.extend(
                [
                    Node::LdInlineAddr(0, RelocAtom::Symbol("label".to_string()).into()),
                    Node::Label("label".to_string(), ()),
                ]
                    .iter()
                    .cloned(),
            )
        })
    }

    fn assert_chunk_size(expected: impl Into<Value>, f: impl FnOnce(&mut Chunk<()>)) {
        let mut object = Object::<()>::new();
        object.add_chunk();
        f(&mut object.chunks[0]);
        let symbols = resolve_symbols(&object);
        assert_eq!(symbols.get(ChunkSize(0)).cloned(), Some(expected.into()))
    }
}
