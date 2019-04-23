use super::{EvalContext, RelocTable, Value};

use crate::model;
use crate::model::{BinOp, ExprOp};
use crate::program::{Atom, Expr, Immediate, NameDef, NameId, SectionId};

use std::borrow::Borrow;

impl<S: Clone> Immediate<S> {
    pub(super) fn eval<R, F>(&self, context: &EvalContext<R, S>, on_undefined: &mut F) -> Value
    where
        R: Borrow<RelocTable>,
        F: FnMut(&S),
    {
        let mut stack = Vec::new();
        for item in &self.0 {
            match &item.op {
                ExprOp::Atom(atom) => stack.push((
                    atom.eval(context, on_undefined).unwrap_or_else(|()| {
                        on_undefined(&item.op_span);
                        Value::Unknown
                    }),
                    item.expr_span.clone(),
                )),
                ExprOp::Binary(operator) => {
                    let rhs = stack.pop().unwrap();
                    let lhs = stack.pop().unwrap();
                    stack.push((operator.apply(&lhs.0, &rhs.0), item.expr_span.clone()))
                }
            }
        }
        stack.pop().unwrap().0
    }
}

impl model::Atom<NameId> {
    fn eval<R, F, S>(&self, context: &EvalContext<R, S>, on_undefined: &mut F) -> Result<Value, ()>
    where
        R: Borrow<RelocTable>,
        F: FnMut(&S),
        S: Clone,
    {
        use model::Atom;
        match self {
            Atom::Const(value) => Ok((*value).into()),
            Atom::LocationCounter => Ok(context.location.clone()),
            Atom::Name(id) => id.eval(context, on_undefined),
            Atom::Param(_) => unimplemented!(),
        }
    }
}

impl<S: Clone> Expr<S> {
    fn eval<R, F>(&self, context: &EvalContext<R, S>, on_undefined: &mut F) -> Value
    where
        R: Borrow<RelocTable>,
        F: FnMut(&S),
    {
        let mut stack = Vec::new();
        for item in &self.0 {
            match &item.op {
                ExprOp::Atom(atom) => stack.push((
                    atom.eval(context, on_undefined).unwrap_or_else(|()| {
                        on_undefined(&item.op_span);
                        Value::Unknown
                    }),
                    item.expr_span.clone(),
                )),
                ExprOp::Binary(operator) => {
                    let rhs = stack.pop().unwrap();
                    let lhs = stack.pop().unwrap();
                    stack.push((operator.apply(&lhs.0, &rhs.0), item.expr_span.clone()))
                }
            }
        }
        stack.pop().unwrap().0
    }
}

impl Atom {
    fn eval<R, F, S>(&self, context: &EvalContext<R, S>, on_undefined: &mut F) -> Result<Value, ()>
    where
        R: Borrow<RelocTable>,
        F: FnMut(&S),
        S: Clone,
    {
        match self {
            Atom::Const(value) => Ok((*value).into()),
            Atom::Name(id) => id.eval(context, on_undefined),
            Atom::Param(_) => unimplemented!(),
            Atom::Reloc(id) => Ok(context.relocs.borrow().get(*id)),
        }
    }
}

impl NameId {
    fn eval<R, F, S>(self, context: &EvalContext<R, S>, on_undefined: &mut F) -> Result<Value, ()>
    where
        R: Borrow<RelocTable>,
        F: FnMut(&S),
        S: Clone,
    {
        let name_def = context.program.names.get(self);
        name_def
            .map(|def| match def {
                NameDef::Section(SectionId(section)) => {
                    let reloc = context.program.sections[*section].addr;
                    context.relocs.borrow().get(reloc)
                }
                NameDef::Symbol(expr) => expr.eval(context, on_undefined),
            })
            .ok_or(())
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
    use crate::model::Atom;
    use crate::program::link::{EvalContext, RelocTable, Value};
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
            Atom::Name(NameId(0)).eval(&context, &mut crate::program::link::ignore_undefined),
            Ok(addr.into())
        )
    }
}
