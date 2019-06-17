use super::{EvalContext, Num, RelocTable};

use crate::model::{Atom, BinOp, Expr, ExprOp, LocationCounter, ParamId};
use crate::program::{BuiltinName, Immediate, NameDef, NameId, RelocId, SectionId};

use std::borrow::Borrow;

impl<S: Clone> Immediate<S> {
    pub(super) fn eval<R, F>(&self, context: &EvalContext<R, S>, on_undefined: &mut F) -> Num
    where
        R: Borrow<RelocTable>,
        F: FnMut(&S),
    {
        self.eval_with_args(context, &[], on_undefined)
    }
}

impl<L, S: Clone> Expr<L, NameId, S> {
    fn eval_with_args<R, F>(
        &self,
        context: &EvalContext<R, S>,
        args: &[SpannedValue<S>],
        on_undefined: &mut F,
    ) -> Num
    where
        R: Borrow<RelocTable>,
        F: FnMut(&S),
        Atom<L, NameId>: Eval,
    {
        let mut stack = Vec::<SpannedValue<S>>::new();
        for item in &self.0 {
            let value = match &item.op {
                ExprOp::Atom(atom) => atom.eval(context, args),
                ExprOp::Binary(operator) => {
                    let rhs = stack.pop().unwrap();
                    let lhs = stack.pop().unwrap().into_num(context, on_undefined);
                    let rhs = rhs.into_num(context, on_undefined);
                    Value::Num(operator.apply(&lhs, &rhs))
                }
                ExprOp::FnCall(n) => {
                    let args = stack.split_off(stack.len() - n);
                    let name = stack.pop().unwrap();
                    let callable = name.value.to_callable(context);
                    callable.call(context, args, on_undefined)
                }
            };
            stack.push(SpannedValue {
                value,
                span: item.expr_span.clone(),
            })
        }
        stack.pop().unwrap().into_num(context, on_undefined)
    }
}

#[derive(Clone)]
struct SpannedValue<S> {
    value: Value,
    span: S,
}

#[derive(Clone)]
enum Value {
    Name(NameId),
    Num(Num),
}

impl<S: Clone> SpannedValue<S> {
    fn into_num<R, F>(self, context: &EvalContext<R, S>, on_undefined: &mut F) -> Num
    where
        R: Borrow<RelocTable>,
        F: FnMut(&S),
    {
        match self.value {
            Value::Name(NameId::Builtin(_)) => unimplemented!(),
            Value::Name(NameId::Def(id)) => match context.program.names.get(id) {
                Some(NameDef::Section(SectionId(section))) => {
                    let reloc = context.program.sections[*section].addr;
                    context.relocs.borrow().get(reloc)
                }
                Some(NameDef::Symbol(expr)) => expr.eval_with_args(context, &[], on_undefined),
                None => {
                    on_undefined(&self.span);
                    Num::Unknown
                }
            },
            Value::Num(value) => value,
        }
    }
}

impl Value {
    fn to_callable<'a, R, S>(&self, context: &'a EvalContext<R, S>) -> Callable<'a, S> {
        match self {
            Value::Name(NameId::Builtin(BuiltinName::Sizeof)) => Callable::Sizeof,
            Value::Name(NameId::Def(id)) => match context.program.names.get(*id) {
                Some(NameDef::Section(_)) => unimplemented!(),
                Some(NameDef::Symbol(expr)) => Callable::Symbol(expr),
                None => unimplemented!(),
            },
            Value::Num(_) => unimplemented!(),
        }
    }
}

enum Callable<'a, S> {
    Sizeof,
    Symbol(&'a Expr<RelocId, NameId, S>),
}

impl<'a, S: Clone> Callable<'a, S> {
    fn call<R, F>(
        &self,
        context: &'a EvalContext<R, S>,
        args: Vec<SpannedValue<S>>,
        on_undefined: &mut F,
    ) -> Value
    where
        R: Borrow<RelocTable>,
        F: FnMut(&S),
    {
        match self {
            Callable::Sizeof => match args.get(0) {
                Some(SpannedValue {
                    value: Value::Name(NameId::Def(id)),
                    ..
                }) => match context.program.names.get(*id) {
                    Some(NameDef::Section(SectionId(section))) => Value::Num(
                        context
                            .relocs
                            .borrow()
                            .get(context.program.sections[*section].size),
                    ),
                    _ => unimplemented!(),
                },
                _ => unimplemented!(),
            },
            Callable::Symbol(expr) => Value::Num(expr.eval_with_args(context, &args, on_undefined)),
        }
    }
}

trait Eval {
    fn eval<R, S>(&self, context: &EvalContext<R, S>, args: &[SpannedValue<S>]) -> Value
    where
        R: Borrow<RelocTable>,
        S: Clone;
}

impl Eval for Atom<LocationCounter, NameId> {
    fn eval<R, S>(&self, context: &EvalContext<R, S>, _: &[SpannedValue<S>]) -> Value
    where
        R: Borrow<RelocTable>,
        S: Clone,
    {
        match self {
            Atom::Const(value) => Value::Num((*value).into()),
            Atom::Location(LocationCounter) => Value::Num(context.location.clone()),
            Atom::Name(id) => Value::Name(*id),
            Atom::Param(_) => unimplemented!(),
        }
    }
}

impl Eval for Atom<RelocId, NameId> {
    fn eval<R, S>(&self, context: &EvalContext<R, S>, args: &[SpannedValue<S>]) -> Value
    where
        R: Borrow<RelocTable>,
        S: Clone,
    {
        match self {
            Atom::Const(value) => Value::Num((*value).into()),
            Atom::Location(id) => Value::Num(context.relocs.borrow().get(*id)),
            Atom::Name(id) => Value::Name(*id),
            Atom::Param(ParamId(id)) => args[*id].value.clone(),
        }
    }
}

impl BinOp {
    fn apply(self, lhs: &Num, rhs: &Num) -> Num {
        match self {
            BinOp::Minus => lhs - rhs,
            BinOp::Multiplication => lhs * rhs,
            BinOp::Plus => lhs + rhs,
            _ => unimplemented!(),
        }
    }
}

pub const BUILTIN_NAMES: &[(&str, NameId)] = &[("sizeof", NameId::Builtin(BuiltinName::Sizeof))];

#[cfg(test)]
mod tests {
    use crate::model::{BinOp, ParamId};
    use crate::program::link::{ignore_undefined, EvalContext, Num, RelocTable};
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
            location: Num::Unknown,
        };
        assert_eq!(
            Immediate::from_atom(NameDefId(0).into(), ()).eval(&context, &mut ignore_undefined),
            addr.into()
        )
    }

    #[test]
    fn eval_section_size() {
        let program = &Program::<()> {
            sections: vec![Section {
                constraints: Constraints { addr: None },
                addr: RelocId(0),
                size: RelocId(1),
                items: vec![],
            }],
            names: NameTable(vec![Some(NameDef::Section(SectionId(0)))]),
            relocs: 2,
        };
        let size = 42;
        let relocs = &RelocTable(vec![0.into(), size.into()]);
        let context = &EvalContext {
            program,
            relocs,
            location: Num::Unknown,
        };
        assert_eq!(
            Immediate::from_items(&[
                BuiltinName::Sizeof.into(),
                NameDefId(0).into(),
                ExprOp::FnCall(1).into()
            ])
            .eval(context, &mut ignore_undefined),
            size.into()
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
            location: Num::Unknown,
        };
        assert_eq!(immediate.eval(context, &mut ignore_undefined), 43.into())
    }
}
