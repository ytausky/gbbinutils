use super::context::{EvalContext, RelocTable};
use super::{BinaryObject, NameDef, NameId, Node, Program, RelocExpr};

use crate::diag::BackendDiagnostics;
use crate::expr::{BinaryOperator, ExprVariant};
use crate::model::{RelocAtom, Width};

use std::borrow::Borrow;
use std::ops::{Add, AddAssign, Mul, RangeInclusive, Sub};

impl<S: Clone> Program<S> {
    pub(crate) fn link(mut self, diagnostics: &mut impl BackendDiagnostics<S>) -> BinaryObject {
        self.resolve_relocs();
        let mut context = EvalContext {
            names: &self.names,
            relocs: &self.relocs,
            location: 0.into(),
        };
        BinaryObject {
            sections: self
                .sections
                .into_iter()
                .map(|section| section.translate(&mut context, diagnostics))
                .collect(),
        }
    }
}

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

impl<S: Clone> Program<S> {
    fn resolve_relocs(&mut self) {
        self.refine_symbols();
        self.refine_symbols();
    }

    fn refine_symbols(&mut self) -> i32 {
        let mut refinements = 0;
        let context = &mut EvalContext {
            names: &self.names,
            relocs: &mut self.relocs,
            location: Value::Unknown,
        };
        for section in &self.sections {
            let (_, size) = section.traverse(context, |item, context| {
                if let Node::Symbol((name, _), expr) = item {
                    let id = match context.names.get_name_def(*name).unwrap() {
                        NameDef::Value(id) => *id,
                    };
                    let value = expr.evaluate(context);
                    refinements += context.relocs.refine(id, value) as i32
                }
            });
            refinements += context.relocs.refine(section.size, size) as i32
        }
        refinements
    }
}

impl<S: Clone> RelocExpr<S> {
    pub(super) fn evaluate<R: Borrow<RelocTable>>(&self, context: &EvalContext<R>) -> Value {
        self.evaluate_strictly(context, &mut |_: &S| ())
    }

    pub(super) fn evaluate_strictly<R, F>(
        &self,
        context: &EvalContext<R>,
        on_undefined_symbol: &mut F,
    ) -> Value
    where
        R: Borrow<RelocTable>,
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

impl RelocAtom<NameId> {
    fn evaluate_strictly<R>(&self, context: &EvalContext<R>) -> Result<Value, ()>
    where
        R: Borrow<RelocTable>,
    {
        use self::RelocAtom::*;
        match self {
            Literal(value) => Ok((*value).into()),
            LocationCounter => Ok(context.location.clone()),
            &Name(id) => {
                let name_def = context.names.get_name_def(id);
                name_def
                    .map(|def| match def {
                        NameDef::Value(id) => context.relocs.borrow().get_value(*id),
                    })
                    .ok_or(())
            }
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

impl<S: Clone> Node<S> {
    pub fn size<R: Borrow<RelocTable>>(&self, context: &EvalContext<R>) -> Value {
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

    use crate::analysis::backend::{AllocName, Backend, PartialBackend};
    use crate::diag::IgnoreDiagnostics;
    use crate::program::context::NameTable;
    use crate::program::{NameDef, ProgramBuilder, Section, ValueId};

    #[test]
    fn resolve_origin_relative_to_previous_section() {
        let origin1 = 0x150;
        let skipped_bytes = 0x10;
        let object = Program {
            sections: vec![
                Section {
                    name: None,
                    addr: Some(origin1.into()),
                    size: ValueId(0),
                    items: vec![Node::Byte(0x42)],
                },
                Section {
                    name: None,
                    addr: Some(
                        ExprVariant::Binary(
                            BinaryOperator::Plus,
                            Box::new(RelocAtom::LocationCounter.into()),
                            Box::new(skipped_bytes.into()),
                        )
                        .into(),
                    ),
                    size: ValueId(1),
                    items: vec![Node::Byte(0x43)],
                },
            ],
            names: NameTable::new(),
            relocs: {
                let mut table = RelocTable::new();
                table.alloc(Value::Unknown);
                table.alloc(Value::Unknown);
                table
            },
        };
        let binary = object.link(&mut IgnoreDiagnostics::new());
        assert_eq!(
            binary.sections[1].addr,
            (origin1 + 1 + skipped_bytes) as usize
        )
    }

    #[test]
    fn label_defined_as_section_origin_plus_offset() {
        let addr = 0xffe1;
        let mut builder = ProgramBuilder::new();
        builder.set_origin(addr.into());
        let symbol_id = builder.alloc_name(());
        builder.define_symbol((symbol_id, ()), RelocAtom::LocationCounter.into());
        let mut object = builder.into_object();
        object.resolve_relocs();
        let value_id = match object.names.get_name_def(symbol_id).unwrap() {
            NameDef::Value(id) => *id,
        };
        assert_eq!(object.relocs.get_value(value_id), addr.into());
    }

    #[test]
    fn empty_section_has_size_zero() {
        assert_section_size(0, |_| ())
    }

    #[test]
    fn section_with_one_byte_has_size_one() {
        assert_section_size(1, |object| object.sections[0].items.push(Node::Byte(0x42)));
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
        assert_section_size(expected, |object| {
            object.sections[0]
                .items
                .push(Node::LdInlineAddr(0, addr.into()))
        });
    }

    #[test]
    fn ld_inline_addr_with_symbol_after_instruction_has_size_three() {
        assert_section_size(3, |object| {
            let name = object.names.alloc_name();
            let value = object.relocs.alloc(Value::Unknown);
            let items = &mut object.sections[0].items;
            items.push(Node::LdInlineAddr(0, RelocAtom::Name(name).into()));
            object.names.define_name(name, NameDef::Value(value));
            items.push(Node::Symbol((name, ()), RelocAtom::LocationCounter.into()))
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

    fn assert_section_size(expected: impl Into<Value>, f: impl FnOnce(&mut Program<()>)) {
        let mut program = Program::new();
        program.add_section(None);
        f(&mut program);
        program.resolve_relocs();
        assert_eq!(
            program.relocs.get_value(program.sections[0].size),
            expected.into()
        )
    }
}
