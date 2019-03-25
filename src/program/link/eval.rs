use super::{EvalContext, RelocTable, Value};

use crate::expr::BinaryOperator;
use crate::model::Atom;
use crate::program::{Expr, NameDef, NameId, SectionId};

use std::borrow::Borrow;

impl<S: Clone> Expr<S> {
    pub(super) fn eval<R, F>(&self, context: &EvalContext<R, S>, on_undefined: &mut F) -> Value
    where
        R: Borrow<RelocTable>,
        F: FnMut(&S),
    {
        use crate::expr::ExprVariant::*;
        match &self.variant {
            Unary(_, _) => unreachable!(),
            Binary(operator, lhs, rhs) => {
                let lhs = lhs.eval(context, on_undefined);
                let rhs = rhs.eval(context, on_undefined);
                operator.apply(&lhs, &rhs)
            }
            Atom(atom) => atom.eval(context).unwrap_or_else(|()| {
                on_undefined(&self.span);
                Value::Unknown
            }),
        }
    }
}

impl Atom<NameId> {
    fn eval<R: Borrow<RelocTable>, S>(&self, context: &EvalContext<R, S>) -> Result<Value, ()> {
        match self {
            &Atom::Attr(id, _attr) => {
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

#[cfg(test)]
mod tests {
    use crate::model::{Atom, Attr};
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
            Atom::Attr(NameId(0), Attr::Addr).eval(&context),
            Ok(addr.into())
        )
    }
}
