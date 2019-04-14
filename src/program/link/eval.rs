use super::{EvalContext, RelocTable, Value};

use crate::model::{Atom, BinOp, ExprOperator};
use crate::program::{Expr, NameDef, NameId, SectionId};

use std::borrow::Borrow;

impl<S: Clone> Expr<S> {
    pub(super) fn eval<R, F>(&self, context: &EvalContext<R, S>, on_undefined: &mut F) -> Value
    where
        R: Borrow<RelocTable>,
        F: FnMut(&S),
    {
        let mut stack = Vec::new();
        for item in &self.0 {
            match &item.op {
                ExprOperator::Atom(atom) => stack.push((
                    atom.eval(context).unwrap_or_else(|()| {
                        on_undefined(&item.op_span);
                        Value::Unknown
                    }),
                    item.expr_span.clone(),
                )),
                ExprOperator::Binary(operator) => {
                    let rhs = stack.pop().unwrap();
                    let lhs = stack.pop().unwrap();
                    stack.push((operator.apply(&lhs.0, &rhs.0), item.expr_span.clone()))
                }
            }
        }
        stack.pop().unwrap().0
    }
}

impl Atom<NameId> {
    fn eval<R: Borrow<RelocTable>, S>(&self, context: &EvalContext<R, S>) -> Result<Value, ()> {
        match self {
            &Atom::Name(id) => {
                let name_def = context.program.names.get(id);
                name_def
                    .map(|def| match def {
                        NameDef::Reloc(id) => context.relocs.borrow().get(*id),
                        NameDef::Section(SectionId(section)) => {
                            let reloc = context.program.sections[*section].addr;
                            context.relocs.borrow().get(reloc)
                        }
                    })
                    .ok_or(())
            }
            Atom::Literal(value) => Ok((*value).into()),
            Atom::LocationCounter => Ok(context.location.clone()),
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
        assert_eq!(Atom::Name(NameId(0)).eval(&context), Ok(addr.into()))
    }
}
