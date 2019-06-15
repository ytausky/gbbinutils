use super::{EvalContext, Num, RelocTable};

use crate::model::{Atom, BinOp, Expr, ExprOp, LocationCounter, ParamId};
use crate::program::{BuiltinName, NameDef, NameId, RelocId, SectionId};

use std::borrow::Borrow;

impl<L, S: Clone> Expr<L, NameId, S> {
    pub(super) fn eval<R, F>(
        &self,
        context: &EvalContext<R, S>,
        args: &[Num],
        on_undefined: &mut F,
    ) -> Num
    where
        R: Borrow<RelocTable>,
        F: FnMut(&S),
        Atom<L, NameId>: Eval,
    {
        let mut stack = Vec::<StackItem<S>>::new();
        for item in &self.0 {
            let variant = match &item.op {
                ExprOp::Atom(atom) => atom.eval(context, args),
                ExprOp::Binary(operator) => {
                    let rhs = stack.pop().unwrap();
                    let lhs = stack.pop().unwrap().into_value(context, on_undefined);
                    let rhs = rhs.into_value(context, on_undefined);
                    StackVariant::Num(operator.apply(&lhs, &rhs))
                }
                ExprOp::FnCall(n) => {
                    let mut args = stack.split_off(stack.len() - n).into_iter();
                    let name = stack.pop().unwrap();
                    match name.variant {
                        StackVariant::Name(NameId::Builtin(BuiltinName::Sizeof)) => {
                            match args.next() {
                                Some(StackItem {
                                    variant: StackVariant::Name(NameId::Def(id)),
                                    ..
                                }) => match context.program.names.get(id) {
                                    Some(NameDef::Section(SectionId(section))) => {
                                        StackVariant::Num(
                                            context
                                                .relocs
                                                .borrow()
                                                .get(context.program.sections[*section].size),
                                        )
                                    }
                                    _ => unimplemented!(),
                                },
                                _ => unimplemented!(),
                            }
                        }
                        StackVariant::Name(NameId::Def(id)) => {
                            let args: Vec<_> = args
                                .map(|arg| arg.into_value(context, on_undefined))
                                .collect();
                            match context.program.names.get(id) {
                                Some(NameDef::Section(_)) => unimplemented!(),
                                Some(NameDef::Symbol(expr)) => {
                                    StackVariant::Num(expr.eval(context, &args, on_undefined))
                                }
                                None => unimplemented!(),
                            }
                        }
                        StackVariant::Num(_) => unimplemented!(),
                    }
                }
            };
            stack.push(StackItem {
                variant,
                span: item.expr_span.clone(),
            })
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
    Num(Num),
}

impl<S: Clone> StackItem<S> {
    fn into_value<R, F>(self, context: &EvalContext<R, S>, on_undefined: &mut F) -> Num
    where
        R: Borrow<RelocTable>,
        F: FnMut(&S),
    {
        match self.variant {
            StackVariant::Name(NameId::Builtin(_)) => unimplemented!(),
            StackVariant::Name(NameId::Def(id)) => match context.program.names.get(id) {
                Some(NameDef::Section(SectionId(section))) => {
                    let reloc = context.program.sections[*section].addr;
                    context.relocs.borrow().get(reloc)
                }
                Some(NameDef::Symbol(expr)) => expr.eval(context, &[], on_undefined),
                None => {
                    on_undefined(&self.span);
                    Num::Unknown
                }
            },
            StackVariant::Num(value) => value,
        }
    }
}

pub(super) trait Eval {
    fn eval<R, S>(&self, context: &EvalContext<R, S>, args: &[Num]) -> StackVariant
    where
        R: Borrow<RelocTable>,
        S: Clone;
}

impl Eval for Atom<LocationCounter, NameId> {
    fn eval<R, S>(&self, context: &EvalContext<R, S>, _: &[Num]) -> StackVariant
    where
        R: Borrow<RelocTable>,
        S: Clone,
    {
        match self {
            Atom::Const(value) => StackVariant::Num((*value).into()),
            Atom::Location(LocationCounter) => StackVariant::Num(context.location.clone()),
            Atom::Name(id) => StackVariant::Name(*id),
            Atom::Param(_) => unimplemented!(),
        }
    }
}

impl Eval for Atom<RelocId, NameId> {
    fn eval<R, S>(&self, context: &EvalContext<R, S>, args: &[Num]) -> StackVariant
    where
        R: Borrow<RelocTable>,
        S: Clone,
    {
        match self {
            Atom::Const(value) => StackVariant::Num((*value).into()),
            Atom::Location(id) => StackVariant::Num(context.relocs.borrow().get(*id)),
            Atom::Name(id) => StackVariant::Name(*id),
            Atom::Param(ParamId(id)) => StackVariant::Num(args[*id].clone()),
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
            Immediate::from_atom(NameDefId(0).into(), ()).eval(
                &context,
                &[],
                &mut ignore_undefined
            ),
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
            .eval(context, &[], &mut ignore_undefined),
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
        assert_eq!(
            immediate.eval(context, &[], &mut ignore_undefined),
            43.into()
        )
    }
}
