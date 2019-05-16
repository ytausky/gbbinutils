use super::{EvalContext, RelocTable, Value};

use crate::model::{Atom, BinOp, Expr, ExprOp, LocationCounter, ParamId};
use crate::program::{NameDef, NameId, RelocId, SectionId};

use std::borrow::Borrow;

impl<L, S: Clone> Expr<L, NameId, S> {
    pub(super) fn eval<R, F>(
        &self,
        context: &EvalContext<R, S>,
        args: &[Value],
        on_undefined: &mut F,
    ) -> Value
    where
        R: Borrow<RelocTable>,
        F: FnMut(&S),
        Atom<L, NameId>: Eval,
    {
        let mut stack = Vec::new();
        for item in &self.0 {
            match &item.op {
                ExprOp::Atom(atom) => stack.push(StackItem {
                    variant: atom.eval(context, args),
                    span: item.expr_span.clone(),
                }),
                ExprOp::Binary(operator) => {
                    let rhs = stack.pop().unwrap();
                    let lhs = stack.pop().unwrap().into_value(context, on_undefined);
                    let rhs = rhs.into_value(context, on_undefined);
                    stack.push(StackItem {
                        variant: StackVariant::Value(operator.apply(&lhs, &rhs)),
                        span: item.expr_span.clone(),
                    })
                }
                ExprOp::FnCall(n) => {
                    let args: Vec<_> = stack
                        .split_off(stack.len() - n)
                        .into_iter()
                        .map(|arg| arg.into_value(context, on_undefined))
                        .collect();
                    let name = stack.pop().unwrap();
                    match name.variant {
                        StackVariant::Name(NameId::Def(id)) => {
                            match context.program.names.get(id) {
                                Some(NameDef::Section(_)) => unimplemented!(),
                                Some(NameDef::Symbol(expr)) => stack.push(StackItem {
                                    variant: StackVariant::Value(expr.eval(
                                        context,
                                        &args,
                                        on_undefined,
                                    )),
                                    span: item.expr_span.clone(),
                                }),
                                None => unimplemented!(),
                            }
                        }
                        StackVariant::Value(_) => unimplemented!(),
                    }
                }
            }
        }
        stack.pop().unwrap().into_value(context, on_undefined)
    }
}

pub(super) struct StackItem<S> {
    variant: StackVariant,
    span: S,
}

pub(super) enum StackVariant {
    Name(NameId),
    Value(Value),
}

impl<S: Clone> StackItem<S> {
    fn into_value<R, F>(self, context: &EvalContext<R, S>, on_undefined: &mut F) -> Value
    where
        R: Borrow<RelocTable>,
        F: FnMut(&S),
    {
        match self.variant {
            StackVariant::Name(NameId::Def(id)) => match context.program.names.get(id) {
                Some(NameDef::Section(SectionId(section))) => {
                    let reloc = context.program.sections[*section].addr;
                    context.relocs.borrow().get(reloc)
                }
                Some(NameDef::Symbol(expr)) => expr.eval(context, &[], on_undefined),
                None => {
                    on_undefined(&self.span);
                    Value::Unknown
                }
            },
            StackVariant::Value(value) => value,
        }
    }
}

pub(super) trait Eval {
    fn eval<R, S>(&self, context: &EvalContext<R, S>, args: &[Value]) -> StackVariant
    where
        R: Borrow<RelocTable>,
        S: Clone;
}

impl Eval for Atom<LocationCounter, NameId> {
    fn eval<R, S>(&self, context: &EvalContext<R, S>, _: &[Value]) -> StackVariant
    where
        R: Borrow<RelocTable>,
        S: Clone,
    {
        match self {
            Atom::Const(value) => StackVariant::Value((*value).into()),
            Atom::Location(LocationCounter) => StackVariant::Value(context.location.clone()),
            Atom::Name(id) => StackVariant::Name(*id),
            Atom::Param(_) => unimplemented!(),
        }
    }
}

impl Eval for Atom<RelocId, NameId> {
    fn eval<R, S>(&self, context: &EvalContext<R, S>, args: &[Value]) -> StackVariant
    where
        R: Borrow<RelocTable>,
        S: Clone,
    {
        match self {
            Atom::Const(value) => StackVariant::Value((*value).into()),
            Atom::Location(id) => StackVariant::Value(context.relocs.borrow().get(*id)),
            Atom::Name(id) => StackVariant::Name(*id),
            Atom::Param(ParamId(id)) => StackVariant::Value(args[*id].clone()),
        }
    }
}

impl BinOp {
    fn apply(self, lhs: &Value, rhs: &Value) -> Value {
        match self {
            BinOp::Minus => lhs - rhs,
            BinOp::Multiplication => lhs * rhs,
            BinOp::Plus => lhs + rhs,
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{BinOp, ParamId};
    use crate::program::link::{ignore_undefined, EvalContext, RelocTable, Value};
    use crate::program::*;

    #[test]
    fn eval_section_addr() {
        let addr = 0x0100;
        let program = Program::<()> {
            sections: vec![Section {
                constraints: Constraints { addr: None },
                addr: RelocId(0),
                size: RelocId(1),
                items: vec![],
            }],
            names: NameTable(vec![Some(NameDef::Section(SectionId(0)))]),
            relocs: 2,
        };
        let relocs = RelocTable(vec![addr.into(), 0.into()]);
        let context = EvalContext {
            program: &program,
            relocs: &relocs,
            location: Value::Unknown,
        };
        assert_eq!(
            Immediate::from_atom(NameDefId(0).into(), ()).eval(
                &context,
                &[],
                &mut ignore_undefined
            ),
            addr.into()
        )
    }

    #[test]
    fn eval_fn_call_in_immediate() {
        let immediate =
            Immediate::from_items(&[NameDefId(0).into(), 42.into(), ExprOp::FnCall(1).into()]);
        let program = &Program::<()> {
            sections: vec![],
            names: NameTable(vec![Some(NameDef::Symbol(Expr::from_items(&[
                ParamId(0).into(),
                1.into(),
                BinOp::Plus.into(),
            ])))]),
            relocs: 0,
        };
        let relocs = &RelocTable(Vec::new());
        let context = &EvalContext {
            program,
            relocs,
            location: Value::Unknown,
        };
        assert_eq!(
            immediate.eval(context, &[], &mut ignore_undefined),
            43.into()
        )
    }
}
