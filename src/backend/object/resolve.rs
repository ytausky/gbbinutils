pub use super::context::{EvalContext, SymbolTable};

use super::context::ChunkSize;
use crate::backend::{Node, Object, RelocAtom, RelocExpr};
use crate::expr::{BinaryOperator, ExprVariant};
use std::borrow::Borrow;
use std::hash::Hash;
use std::ops::{Add, AddAssign, Mul, RangeInclusive, Sub};

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

impl From<RangeInclusive<i32>> for Value {
    fn from(range: RangeInclusive<i32>) -> Self {
        Value::Range {
            min: *range.start(),
            max: *range.end(),
        }
    }
}

impl AddAssign<&Value> for Value {
    fn add_assign(&mut self, rhs: &Value) {
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

impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: &Value) -> Self::Output {
        let mut result = self.clone();
        result += rhs;
        result
    }
}

impl Sub for &Value {
    type Output = Value;
    fn sub(self, rhs: &Value) -> Self::Output {
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

impl Mul for &Value {
    type Output = Value;

    fn mul(self, rhs: &Value) -> Self::Output {
        match (self, rhs) {
            (
                Value::Range {
                    min: lhs_min,
                    max: lhs_max,
                },
                Value::Range {
                    min: rhs_min,
                    max: rhs_max,
                },
            ) => {
                let lhs_endpoints = [lhs_min, lhs_max];
                let rhs_endpoints = [rhs_min, rhs_max];
                let products = lhs_endpoints
                    .iter()
                    .cloned()
                    .flat_map(|n| rhs_endpoints.iter().map(move |&m| m * n));
                Value::Range {
                    min: products.clone().min().unwrap(),
                    max: products.max().unwrap(),
                }
            }
            _ => Value::Unknown,
        }
    }
}

impl<I: Clone + Eq + Hash, S: Clone> Object<I, S> {
    pub fn resolve_symbols(&mut self) {
        self.collect_symbols();
        self.refine_symbols();
    }

    fn collect_symbols(&mut self) {
        (0..self.chunks.len()).for_each(|i| self.symbols.define(ChunkSize(i), Value::Unknown));
        {
            let mut context = EvalContext {
                symbols: &mut self.symbols,
                location: Value::Unknown,
            };
            for (i, chunk) in self.chunks.iter().enumerate() {
                let size = chunk.traverse(&mut context, |item, context| {
                    if let Node::Symbol((symbol, _), expr) = item {
                        let value = expr.evaluate(context);
                        context.symbols.define(symbol.clone(), value)
                    }
                });
                context.symbols.refine(&ChunkSize(i), size);
            }
        }
    }

    fn refine_symbols(&mut self) -> i32 {
        let mut refinements = 0;
        let context = &mut EvalContext {
            symbols: &mut self.symbols,
            location: Value::Unknown,
        };
        for (i, chunk) in self.chunks.iter().enumerate() {
            let size = chunk.traverse(context, |item, context| {
                if let Node::Symbol((symbol, _), expr) = item {
                    let value = expr.evaluate(context);
                    refinements += context.symbols.refine(symbol, value) as i32
                }
            });
            refinements += context.symbols.refine(&ChunkSize(i), size) as i32
        }
        refinements
    }
}

impl<I: Eq + Hash, S: Clone> RelocExpr<I, S> {
    pub fn evaluate<ST: Borrow<SymbolTable<I>>>(&self, context: &EvalContext<ST>) -> Value {
        self.evaluate_strictly(context, &mut |_: &S| ())
    }

    pub fn evaluate_strictly<ST, F>(
        &self,
        context: &EvalContext<ST>,
        on_undefined_symbol: &mut F,
    ) -> Value
    where
        ST: Borrow<SymbolTable<I>>,
        F: FnMut(&S),
    {
        use self::ExprVariant::*;
        match &self.variant {
            Unary(_, _) => unreachable!(),
            Binary(operator, lhs, rhs) => {
                let lhs = lhs.evaluate_strictly(context, on_undefined_symbol);
                let rhs = rhs.evaluate_strictly(context, on_undefined_symbol);
                operator.apply(&lhs, &rhs)
            }
            Atom(atom) => atom.evaluate_strictly(context).unwrap_or_else(|()| {
                on_undefined_symbol(&self.span);
                Value::Unknown
            }),
        }
    }
}

impl<I: Eq + Hash> RelocAtom<I> {
    fn evaluate_strictly<ST>(&self, context: &EvalContext<ST>) -> Result<Value, ()>
    where
        ST: Borrow<SymbolTable<I>>,
    {
        use self::RelocAtom::*;
        match self {
            Literal(value) => Ok((*value).into()),
            LocationCounter => Ok(context.location.clone()),
            Symbol(symbol) => context.symbols.borrow().get(symbol).cloned().ok_or(()),
        }
    }
}

impl BinaryOperator {
    fn apply(self, lhs: &Value, rhs: &Value) -> Value {
        match self {
            BinaryOperator::Minus => lhs - rhs,
            BinaryOperator::Multiplication => lhs * rhs,
            BinaryOperator::Plus => lhs + rhs,
            _ => unimplemented!(),
        }
    }
}

impl<I: Eq + Hash, S: Clone> Node<I, S> {
    pub fn size<ST: Borrow<SymbolTable<I>>>(&self, context: &EvalContext<ST>) -> Value {
        match self {
            Node::Byte(_) | Node::Embedded(..) => 1.into(),
            Node::Expr(_, width) => width.len().into(),
            Node::LdInlineAddr(_, expr) => match expr.evaluate(context) {
                Value::Range { min, .. } if min >= 0xff00 => 2.into(),
                Value::Range { max, .. } if max < 0xff00 => 3.into(),
                _ => Value::Range { min: 2, max: 3 },
            },
            Node::Symbol(..) => 0.into(),
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
        builder.define_symbol((label.into(), ()), RelocAtom::LocationCounter.into());
        let mut object = builder.into_object();
        object.resolve_symbols();
        assert_eq!(object.symbols.get(label), Some(&addr.into()))
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
                    Node::Symbol(("label".to_string(), ()), RelocAtom::LocationCounter.into()),
                ]
                .iter()
                .cloned(),
            )
        })
    }

    macro_rules! triples {
        ($(($lhs:expr, $rhs:expr, $result:expr)),*) => {
            [$(($lhs.into(), $rhs.into(), $result.into())),*]
        };
    }

    #[test]
    fn multiply_ranges() {
        let cases: &[(Value, Value, Value)] = &triples![
            (2, 3, 6),
            (-1..=1, -1..=1, -1..=1),
            (1..=6, -1, -6..=-1),
            (-6..=1, -6..=1, -6..=36)
        ];
        for (lhs, rhs, product) in cases {
            assert_eq!(BinaryOperator::Multiplication.apply(lhs, rhs), *product)
        }
    }

    fn assert_chunk_size(expected: impl Into<Value>, f: impl FnOnce(&mut Chunk<String, ()>)) {
        let mut object = Object::<String, ()>::new();
        object.add_chunk();
        f(&mut object.chunks[0]);
        object.resolve_symbols();
        assert_eq!(
            object.symbols.get(&ChunkSize(0)).cloned(),
            Some(expected.into())
        )
    }
}
