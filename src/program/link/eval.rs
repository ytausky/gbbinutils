use super::{EvalContext, Num, RelocTable};

use crate::diag::span::{Spanned, WithSpan};
use crate::diag::{BackendDiagnostics, Message};
use crate::model::{Atom, BinOp, Expr, ExprOp, LocationCounter, ParamId};
use crate::program::{BuiltinName, Immediate, NameDef, NameDefId, NameId, RelocId, SectionId};

use std::borrow::Borrow;

impl<S: Clone> Immediate<S> {
    pub(super) fn to_num<R, D>(&self, context: &EvalContext<R, S>, diagnostics: &mut D) -> Num
    where
        R: Borrow<RelocTable>,
        D: BackendDiagnostics<S>,
    {
        self.eval(context, &[], diagnostics)
    }
}

trait Eval<'a, S: Clone> {
    type Output;

    fn eval<R, D>(
        self,
        context: &'a EvalContext<R, S>,
        args: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output
    where
        R: Borrow<RelocTable>,
        D: BackendDiagnostics<S>;
}

#[derive(Clone)]
enum Value<'a, S: Clone> {
    Name(ResolvedName<'a, S>),
    Num(Num),
    Unresolved,
}

impl<'a, L, S: Clone> Eval<'a, S> for &'a Expr<L, NameId, S>
where
    for<'r> Spanned<&'r Atom<L, NameId>, &'r S>: Eval<'a, S, Output = Value<'a, S>>,
{
    type Output = Num;

    fn eval<R, D>(
        self,
        context: &'a EvalContext<R, S>,
        args: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Num
    where
        R: Borrow<RelocTable>,
        D: BackendDiagnostics<S>,
    {
        let mut stack = Vec::<Spanned<Value<_>, _>>::new();
        for item in &self.0 {
            let value = match &item.op {
                ExprOp::Atom(atom) => {
                    atom.with_span(&item.expr_span)
                        .eval(context, args, diagnostics)
                }
                ExprOp::Binary(operator) => {
                    let rhs = stack.pop().unwrap();
                    let lhs = stack.pop().unwrap().eval(context, &[], diagnostics);
                    let rhs = rhs.eval(context, &[], diagnostics);
                    Value::Num(operator.apply(&lhs, &rhs))
                }
                ExprOp::FnCall(n) => {
                    let args = stack.split_off(stack.len() - n);
                    let name = stack.pop().unwrap();
                    Value::Num(name.eval(context, &args, diagnostics))
                }
            };
            stack.push(value.with_span(&item.expr_span))
        }
        stack.pop().unwrap().eval(context, &[], diagnostics)
    }
}

impl<'a, S: Clone> Eval<'a, S> for Spanned<Value<'a, S>, &S> {
    type Output = Num;

    fn eval<R, D>(
        self,
        context: &'a EvalContext<R, S>,
        args: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output
    where
        R: Borrow<RelocTable>,
        D: BackendDiagnostics<S>,
    {
        match self.item {
            Value::Name(name) => name.with_span(self.span).eval(context, args, diagnostics),
            Value::Num(value) => value,
            Value::Unresolved => Num::Unknown,
        }
    }
}

impl<S: Clone> Spanned<NameId, &S> {
    fn to_value<'a, R, D: BackendDiagnostics<S>>(
        &self,
        context: &'a EvalContext<R, S>,
        diagnostics: &mut D,
    ) -> Value<'a, S> {
        self.resolve(context, diagnostics)
            .map(Value::Name)
            .unwrap_or(Value::Unresolved)
    }

    fn resolve<'a, R, D: BackendDiagnostics<S>>(
        &self,
        context: &'a EvalContext<R, S>,
        diagnostics: &mut D,
    ) -> Option<ResolvedName<'a, S>> {
        match self.item {
            NameId::Builtin(BuiltinName::Sizeof) => Some(ResolvedName::Sizeof),
            NameId::Def(id) => id.with_span(self.span).resolve(context, diagnostics),
        }
    }
}

impl<S: Clone> Spanned<NameDefId, &S> {
    fn resolve<'a, R, D: BackendDiagnostics<S>>(
        &self,
        context: &'a EvalContext<R, S>,
        diagnostics: &mut D,
    ) -> Option<ResolvedName<'a, S>> {
        let id = self.item;
        let resolved = context.program.names.get(id).map(ResolvedName::NameDef);
        if resolved.is_none() {
            let symbol = diagnostics.strip_span(self.span);
            diagnostics.emit_diag(Message::UnresolvedSymbol { symbol }.at(self.span.clone()))
        }
        resolved
    }
}

#[derive(Clone)]
enum ResolvedName<'a, S> {
    NameDef(&'a NameDef<S>),
    Sizeof,
}

impl<'a, S: Clone> Eval<'a, S> for Spanned<ResolvedName<'a, S>, &S> {
    type Output = Num;

    fn eval<R, D>(
        self,
        context: &'a EvalContext<R, S>,
        args: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output
    where
        R: Borrow<RelocTable>,
        D: BackendDiagnostics<S>,
    {
        match self.item {
            ResolvedName::NameDef(def) => def.eval(context, args, diagnostics),
            ResolvedName::Sizeof => match args.get(0) {
                Some(Spanned {
                    item: Value::Name(ResolvedName::NameDef(NameDef::Section(SectionId(section)))),
                    ..
                }) => context
                    .relocs
                    .borrow()
                    .get(context.program.sections[*section].size),
                None => {
                    let name = diagnostics.strip_span(self.span);
                    diagnostics.emit_diag(
                        Message::CannotCoerceBuiltinNameIntoNum { name }.at(self.span.clone()),
                    );
                    Num::Unknown
                }
                _ => unimplemented!(),
            },
        }
    }
}

impl<'a, S: Clone> Eval<'a, S> for &'a NameDef<S> {
    type Output = Num;

    fn eval<R, D>(
        self,
        context: &'a EvalContext<R, S>,
        args: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output
    where
        R: Borrow<RelocTable>,
        D: BackendDiagnostics<S>,
    {
        match self {
            NameDef::Section(SectionId(section)) => context
                .relocs
                .borrow()
                .get(context.program.sections[*section].addr),
            NameDef::Symbol(expr) => expr.eval(context, args, diagnostics),
        }
    }
}

impl<'a, S: Clone + 'a> Eval<'a, S> for Spanned<&Atom<LocationCounter, NameId>, &S> {
    type Output = Value<'a, S>;

    fn eval<R, D>(
        self,
        context: &'a EvalContext<R, S>,
        _: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output
    where
        R: Borrow<RelocTable>,
        D: BackendDiagnostics<S>,
    {
        match self.item {
            Atom::Const(value) => Value::Num((*value).into()),
            Atom::Location(LocationCounter) => Value::Num(context.location.clone()),
            Atom::Name(id) => (*id).with_span(self.span).to_value(context, diagnostics),
            Atom::Param(_) => unimplemented!(),
        }
    }
}

impl<'a, S: Clone + 'a> Eval<'a, S> for Spanned<&Atom<RelocId, NameId>, &S> {
    type Output = Value<'a, S>;

    fn eval<R, D>(
        self,
        context: &'a EvalContext<R, S>,
        args: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output
    where
        R: Borrow<RelocTable>,
        D: BackendDiagnostics<S>,
    {
        match self.item {
            Atom::Const(value) => Value::Num((*value).into()),
            Atom::Location(id) => Value::Num(context.relocs.borrow().get(*id)),
            Atom::Name(id) => (*id).with_span(self.span).to_value(context, diagnostics),
            Atom::Param(ParamId(id)) => args[*id].item.clone(),
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
    use super::{BinOp, EvalContext, Expr, Immediate, Message, Num, ParamId, RelocTable};

    use crate::diag::{DiagnosticsEvent, IgnoreDiagnostics, Merge, MockDiagnostics, MockSpan};
    use crate::log::Log;
    use crate::model::ExprItem;
    use crate::program::*;

    #[test]
    fn eval_section_addr() {
        let addr = 0x0100;
        let program = &mk_program_with_empty_section();
        let relocs = &RelocTable(vec![addr.into(), 0.into()]);
        let context = EvalContext {
            program,
            relocs,
            location: Num::Unknown,
        };
        assert_eq!(
            Immediate::from_atom(NameDefId(0).into(), ()).to_num(&context, &mut IgnoreDiagnostics),
            addr.into()
        )
    }

    #[test]
    fn eval_section_size() {
        let program = &mk_program_with_empty_section();
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
            .to_num(context, &mut IgnoreDiagnostics),
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
        assert_eq!(immediate.to_num(context, &mut IgnoreDiagnostics), 43.into())
    }

    #[test]
    fn eval_section_name_call() {
        let addr = 0x1337;
        let program = &mk_program_with_empty_section();
        let relocs = &RelocTable(vec![addr.into(), 0.into()]);
        let context = EvalContext {
            program,
            relocs,
            location: Num::Unknown,
        };
        let immediate = Immediate::from_items(&[NameDefId(0).into(), ExprOp::FnCall(0).into()]);
        assert_eq!(
            immediate.to_num(&context, &mut IgnoreDiagnostics),
            addr.into()
        )
    }

    #[test]
    fn diagnose_using_sizeof_as_immediate() {
        let program = &Program {
            sections: vec![],
            names: NameTable(vec![]),
            relocs: 0,
        };
        let relocs = &RelocTable(vec![]);
        let context = &EvalContext {
            program,
            relocs,
            location: Num::Unknown,
        };
        let mut diagnostics = MockDiagnostics::<DiagnosticsEvent<MockSpan<_>>>::new(Log::new());
        let immediate = Immediate::from_atom(
            Atom::Name(NameId::Builtin(BuiltinName::Sizeof)),
            MockSpan::from(0),
        );
        let value = immediate.to_num(context, &mut diagnostics);
        let log = diagnostics.into_log();
        assert_eq!(value, Num::Unknown);
        assert_eq!(
            log,
            [DiagnosticsEvent::EmitDiag(
                Message::CannotCoerceBuiltinNameIntoNum { name: 0.into() }
                    .at(0.into())
                    .into()
            )]
        )
    }

    #[test]
    fn diagnose_calling_undefined_symbol() {
        let program = &Program {
            sections: vec![],
            names: NameTable(vec![None]),
            relocs: 0,
        };
        let relocs = &RelocTable::new(0);
        let context = EvalContext {
            program,
            relocs,
            location: Num::Unknown,
        };
        let mut diagnostics = MockDiagnostics::<DiagnosticsEvent<_>>::new(Log::new());
        let name_span = MockSpan::from("name");
        let call_span = MockSpan::from("call");
        let immediate = Expr(vec![
            ExprItem {
                op: ExprOp::Atom(Atom::Name(NameId::Def(NameDefId(0)))),
                op_span: name_span.clone(),
                expr_span: name_span.clone(),
            },
            ExprItem {
                op: ExprOp::FnCall(0),
                op_span: call_span.clone(),
                expr_span: MockSpan::merge(name_span.clone(), call_span),
            },
        ]);
        let value = immediate.to_num(&context, &mut diagnostics);
        assert_eq!(value, Num::Unknown);
        assert_eq!(
            diagnostics.into_log(),
            [DiagnosticsEvent::EmitDiag(
                Message::UnresolvedSymbol {
                    symbol: name_span.clone()
                }
                .at(name_span)
                .into()
            )]
        )
    }

    fn mk_program_with_empty_section<S>() -> Program<S> {
        Program {
            sections: vec![Section {
                constraints: Constraints { addr: None },
                addr: RelocId(0),
                size: RelocId(1),
                items: vec![],
            }],
            names: NameTable(vec![Some(NameDef::Section(SectionId(0)))]),
            relocs: 2,
        }
    }
}
