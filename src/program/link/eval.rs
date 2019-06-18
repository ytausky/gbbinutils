use super::{EvalContext, Num, RelocTable};

use crate::diag::span::StripSpan;
use crate::diag::{BackendDiagnostics, EmitDiag, Message};
use crate::model::{Atom, BinOp, Expr, ExprOp, LocationCounter, ParamId};
use crate::program::{BuiltinName, Immediate, NameDef, NameDefId, NameId, RelocId, SectionId};

use std::borrow::Borrow;

impl<S: Clone> Immediate<S> {
    pub(super) fn eval<R, D>(&self, context: &EvalContext<R, S>, diagnostics: &mut D) -> Num
    where
        R: Borrow<RelocTable>,
        D: EvalDiagnostics<S>,
    {
        self.eval_with_args(context, &[], diagnostics)
    }
}

pub(super) trait EvalDiagnostics<S> {
    type Diagnostics: BackendDiagnostics<S>;

    fn with_diag(&mut self, f: impl FnOnce(&mut Self::Diagnostics));
}

impl<D: BackendDiagnostics<S>, S: Clone> EvalDiagnostics<S> for D {
    type Diagnostics = Self;

    fn with_diag(&mut self, f: impl FnOnce(&mut Self::Diagnostics)) {
        f(self)
    }
}

impl<L, S: Clone> Expr<L, NameId, S> {
    fn eval_with_args<R, D>(
        &self,
        context: &EvalContext<R, S>,
        args: &[SpannedValue<S>],
        diagnostics: &mut D,
    ) -> Num
    where
        R: Borrow<RelocTable>,
        D: EvalDiagnostics<S>,
        Atom<L, NameId>: Eval<S>,
    {
        let mut stack = Vec::<SpannedValue<S>>::new();
        for item in &self.0 {
            let value = match &item.op {
                ExprOp::Atom(atom) => atom.eval(item.expr_span.clone(), context, args, diagnostics),
                ExprOp::Binary(operator) => {
                    let rhs = stack.pop().unwrap();
                    let lhs = stack.pop().unwrap().eval(context, vec![], diagnostics);
                    let rhs = rhs.eval(context, vec![], diagnostics);
                    Value::Num(operator.apply(&lhs, &rhs))
                }
                ExprOp::FnCall(n) => {
                    let args = stack.split_off(stack.len() - n);
                    let name = stack.pop().unwrap();
                    Value::Num(name.eval(context, args, diagnostics))
                }
            };
            stack.push(SpannedValue {
                value,
                span: item.expr_span.clone(),
            })
        }
        stack.pop().unwrap().eval(context, vec![], diagnostics)
    }
}

struct SpannedValue<'a, S: Clone> {
    value: Value<'a, S>,
    span: S,
}

#[derive(Clone)]
enum Value<'a, S: Clone> {
    Name(ResolvedName<'a, S>),
    Num(Num),
    Unresolved,
}

impl<'a, S: Clone> SpannedValue<'a, S> {
    fn eval<R, D>(
        self,
        context: &EvalContext<R, S>,
        args: Vec<SpannedValue<S>>,
        diagnostics: &mut D,
    ) -> Num
    where
        R: Borrow<RelocTable>,
        D: EvalDiagnostics<S>,
    {
        match self.value {
            Value::Name(name) => name.eval(self.span, context, args, diagnostics),
            Value::Num(value) => value,
            Value::Unresolved => Num::Unknown,
        }
    }
}

impl NameId {
    fn resolve<'a, R, D, S>(
        &self,
        span: S,
        context: &'a EvalContext<R, S>,
        diagnostics: &mut D,
    ) -> Option<ResolvedName<'a, S>>
    where
        D: EvalDiagnostics<S>,
    {
        match self {
            NameId::Builtin(BuiltinName::Sizeof) => Some(ResolvedName::Sizeof),
            NameId::Def(id) => id.resolve(span, context, diagnostics),
        }
    }
}

impl NameDefId {
    fn resolve<'a, R, D, S>(
        self,
        span: S,
        context: &'a EvalContext<R, S>,
        diagnostics: &mut D,
    ) -> Option<ResolvedName<'a, S>>
    where
        D: EvalDiagnostics<S>,
    {
        let resolved = context.program.names.get(self).map(ResolvedName::NameDef);
        if resolved.is_none() {
            diagnostics.with_diag(|diagnostics| {
                let symbol = diagnostics.strip_span(&span);
                diagnostics.emit_diag(Message::UnresolvedSymbol { symbol }.at(span))
            });
        }
        resolved
    }
}

#[derive(Clone)]
enum ResolvedName<'a, S> {
    NameDef(&'a NameDef<S>),
    Sizeof,
}

impl<'a, S: Clone> ResolvedName<'a, S> {
    fn eval<R, D>(
        &self,
        span: S,
        context: &'a EvalContext<R, S>,
        args: Vec<SpannedValue<S>>,
        diagnostics: &mut D,
    ) -> Num
    where
        R: Borrow<RelocTable>,
        D: EvalDiagnostics<S>,
    {
        match self {
            ResolvedName::NameDef(def) => def.eval(context, args, diagnostics),
            ResolvedName::Sizeof => match args.get(0) {
                Some(SpannedValue {
                    value: Value::Name(ResolvedName::NameDef(NameDef::Section(SectionId(section)))),
                    ..
                }) => context
                    .relocs
                    .borrow()
                    .get(context.program.sections[*section].size),
                None => {
                    diagnostics.with_diag(|diagnostics| {
                        let name = diagnostics.strip_span(&span);
                        diagnostics
                            .emit_diag(Message::CannotCoerceBuiltinNameIntoNum { name }.at(span));
                    });
                    Num::Unknown
                }
                _ => unimplemented!(),
            },
        }
    }
}

impl<S: Clone> NameDef<S> {
    fn eval<R, D>(
        &self,
        context: &EvalContext<R, S>,
        args: Vec<SpannedValue<S>>,
        diagnostics: &mut D,
    ) -> Num
    where
        R: Borrow<RelocTable>,
        D: EvalDiagnostics<S>,
    {
        match self {
            NameDef::Section(SectionId(section)) => context
                .relocs
                .borrow()
                .get(context.program.sections[*section].addr),
            NameDef::Symbol(expr) => expr.eval_with_args(context, &args, diagnostics),
        }
    }
}

trait Eval<S: Clone> {
    fn eval<'a, R, D>(
        &self,
        span: S,
        context: &'a EvalContext<R, S>,
        args: &'a [SpannedValue<S>],
        diagnostics: &mut D,
    ) -> Value<'a, S>
    where
        R: Borrow<RelocTable>,
        D: EvalDiagnostics<S>;
}

impl<S: Clone> Eval<S> for Atom<LocationCounter, NameId> {
    fn eval<'a, R, D>(
        &self,
        span: S,
        context: &'a EvalContext<R, S>,
        _: &'a [SpannedValue<S>],
        diagnostics: &mut D,
    ) -> Value<'a, S>
    where
        R: Borrow<RelocTable>,
        D: EvalDiagnostics<S>,
    {
        match self {
            Atom::Const(value) => Value::Num((*value).into()),
            Atom::Location(LocationCounter) => Value::Num(context.location.clone()),
            Atom::Name(id) => id
                .resolve(span, context, diagnostics)
                .map(Value::Name)
                .unwrap_or(Value::Unresolved),
            Atom::Param(_) => unimplemented!(),
        }
    }
}

impl<S: Clone> Eval<S> for Atom<RelocId, NameId> {
    fn eval<'a, R, D>(
        &self,
        span: S,
        context: &'a EvalContext<R, S>,
        args: &'a [SpannedValue<S>],
        diagnostics: &mut D,
    ) -> Value<'a, S>
    where
        R: Borrow<RelocTable>,
        D: EvalDiagnostics<S>,
    {
        match self {
            Atom::Const(value) => Value::Num((*value).into()),
            Atom::Location(id) => Value::Num(context.relocs.borrow().get(*id)),
            Atom::Name(id) => id
                .resolve(span, context, diagnostics)
                .map(Value::Name)
                .unwrap_or(Value::Unresolved),
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
            Immediate::from_atom(NameDefId(0).into(), ()).eval(&context, &mut IgnoreDiagnostics),
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
            .eval(context, &mut IgnoreDiagnostics),
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
        assert_eq!(immediate.eval(context, &mut IgnoreDiagnostics), 43.into())
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
            immediate.eval(&context, &mut IgnoreDiagnostics),
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
        let value = immediate.eval(context, &mut diagnostics);
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
        let value = immediate.eval(&context, &mut diagnostics);
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
