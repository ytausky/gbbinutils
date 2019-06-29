use super::{EvalContext, Num, RelocTable};

use crate::diag::span::{Spanned, WithSpan};
use crate::diag::{BackendDiagnostics, Message, ValueKind};
use crate::model::{Atom, BinOp, Expr, ExprOp, LocationCounter, ParamId};
use crate::program::*;

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

    fn eval<R: Borrow<RelocTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a EvalContext<R, S>,
        args: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output;
}

#[derive(Clone)]
enum Value<'a, S: Clone> {
    Name(ResolvedName<'a, S>),
    Num(Num),
    Unresolved,
}

#[derive(Clone)]
enum ResolvedName<'a, S> {
    Section(&'a Section<S>),
    Sizeof,
    Symbol(&'a Expr<Atom<RelocId, NameId>, S>),
}

impl<'a, L, S: Clone> Eval<'a, S> for &'a Expr<Atom<L, NameId>, S>
where
    for<'r> Spanned<&'r Atom<L, NameId>, &'r S>: Eval<'a, S, Output = Value<'a, S>>,
{
    type Output = Num;

    fn eval<R: Borrow<RelocTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a EvalContext<R, S>,
        args: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output {
        let mut stack = Vec::<Spanned<Value<_>, _>>::new();
        for Spanned { item, span } in &self.0 {
            let value = match item {
                ExprOp::Atom(atom) => atom.with_span(span).eval(context, args, diagnostics),
                ExprOp::Binary(operator) => {
                    let rhs = stack.pop().unwrap();
                    let lhs = stack.pop().unwrap().eval(context, &[], diagnostics);
                    let rhs = rhs.eval(context, &[], diagnostics);
                    Value::Num(operator.apply(&lhs, &rhs))
                }
                ExprOp::FnCall(n) => {
                    let name = stack.pop().unwrap();
                    let arg_index = stack.len() - n;
                    let value = Value::Num(name.eval(context, &stack[arg_index..], diagnostics));
                    stack.truncate(arg_index);
                    value
                }
            };
            stack.push(value.with_span(span))
        }
        stack.pop().unwrap().eval(context, &[], diagnostics)
    }
}

impl<'a, S: Clone> Eval<'a, S> for Spanned<Value<'a, S>, &S> {
    type Output = Num;

    fn eval<R: Borrow<RelocTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a EvalContext<R, S>,
        args: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output {
        match self.item {
            Value::Name(name) => name.with_span(self.span).eval(context, args, diagnostics),
            Value::Num(value) => value,
            Value::Unresolved => Num::Unknown,
        }
    }
}

impl<'a, S: Clone> Eval<'a, S> for Spanned<ResolvedName<'a, S>, &S> {
    type Output = Num;

    fn eval<R: Borrow<RelocTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a EvalContext<R, S>,
        args: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output {
        match self.item {
            ResolvedName::Section(section) => context.relocs.borrow().get(section.addr),
            ResolvedName::Sizeof => args
                .get(0)
                .map(|value| value.sizeof(context, diagnostics))
                .unwrap_or_else(|| {
                    let name = diagnostics.strip_span(self.span);
                    diagnostics.emit_diag(
                        Message::CannotCoerceBuiltinNameIntoNum { name }.at(self.span.clone()),
                    );
                    Num::Unknown
                }),
            ResolvedName::Symbol(expr) => expr.eval(context, args, diagnostics),
        }
    }
}

impl<'a, S: Clone + 'a> Eval<'a, S> for Spanned<&Atom<LocationCounter, NameId>, &S> {
    type Output = Value<'a, S>;

    fn eval<R: Borrow<RelocTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a EvalContext<R, S>,
        _: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output {
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

    fn eval<R: Borrow<RelocTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a EvalContext<R, S>,
        args: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output {
        match self.item {
            Atom::Const(value) => Value::Num((*value).into()),
            Atom::Location(id) => Value::Num(context.relocs.borrow().get(*id)),
            Atom::Name(id) => (*id).with_span(self.span).to_value(context, diagnostics),
            Atom::Param(ParamId(id)) => args[*id].item.clone(),
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
        let resolved = context.program.names.get(id).map(|def| match def {
            NameDef::Section(SectionId(id)) => {
                ResolvedName::Section(&context.program.sections[*id])
            }
            NameDef::Symbol(expr) => ResolvedName::Symbol(expr),
        });
        if resolved.is_none() {
            let symbol = diagnostics.strip_span(self.span);
            diagnostics.emit_diag(Message::UnresolvedSymbol { symbol }.at(self.span.clone()))
        }
        resolved
    }
}

impl BinOp {
    fn apply(self, lhs: &Num, rhs: &Num) -> Num {
        match self {
            BinOp::BitwiseOr => lhs | rhs,
            BinOp::Division => unimplemented!(),
            BinOp::Minus => lhs - rhs,
            BinOp::Multiplication => lhs * rhs,
            BinOp::Plus => lhs + rhs,
        }
    }
}

impl<'a, S: Clone> Spanned<Value<'a, S>, &S> {
    fn sizeof<R, D>(&self, context: &'a EvalContext<R, S>, diagnostics: &mut D) -> Num
    where
        R: Borrow<RelocTable>,
        D: BackendDiagnostics<S>,
    {
        match self.item {
            Value::Name(ResolvedName::Section(section)) => {
                context.relocs.borrow().get(section.size)
            }
            ref other => {
                if let Some(found) = other.kind() {
                    diagnostics.emit_diag(
                        Message::ExpectedFound {
                            expected: ValueKind::Section,
                            found,
                        }
                        .at(self.span.clone()),
                    )
                }
                Num::Unknown
            }
        }
    }
}

impl<'a, S: Clone> Value<'a, S> {
    fn kind(&self) -> Option<ValueKind> {
        match self {
            Value::Name(ResolvedName::Section(_)) => Some(ValueKind::Section),
            Value::Name(ResolvedName::Sizeof) => Some(ValueKind::Builtin),
            Value::Name(ResolvedName::Symbol(_)) => Some(ValueKind::Symbol),
            Value::Num(_) => Some(ValueKind::Num),
            Value::Unresolved => None,
        }
    }
}

pub const BUILTIN_NAMES: &[(&str, NameId)] = &[("sizeof", NameId::Builtin(BuiltinName::Sizeof))];

#[cfg(test)]
mod tests {
    use super::*;

    use crate::diag::*;
    use crate::log::Log;

    type MockDiagnostics<S> = crate::diag::MockDiagnostics<DiagnosticsEvent<S>>;

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
                NameDefId(0).into(),
                BuiltinName::Sizeof.into(),
                ExprOp::FnCall(1).into()
            ])
            .to_num(context, &mut IgnoreDiagnostics),
            size.into()
        )
    }

    #[test]
    fn eval_fn_call_in_immediate() {
        let immediate =
            Immediate::from_items(&[42.into(), NameDefId(0).into(), ExprOp::FnCall(1).into()]);
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
    fn eval_bitwise_or() {
        assert_eq!(
            eval_in_empty_program(
                Immediate::from_items(&[
                    0x17.into(),
                    0x86.into(),
                    ExprOp::Binary(BinOp::BitwiseOr).into(),
                ]),
                &mut IgnoreDiagnostics
            ),
            0x97.into()
        )
    }

    #[test]
    fn diagnose_using_sizeof_as_immediate() {
        let mut diagnostics = MockDiagnostics::new(Log::new());
        let immediate = Immediate::from_atom(
            Atom::Name(NameId::Builtin(BuiltinName::Sizeof)),
            MockSpan::from(0),
        );
        let value = eval_in_empty_program(immediate, &mut diagnostics);
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
        let mut diagnostics = MockDiagnostics::new(Log::new());
        let name_span = MockSpan::from("name");
        let call_span = MockSpan::from("call");
        let immediate = Expr(vec![
            ExprOp::Atom(Atom::Name(NameId::Def(NameDefId(0)))).with_span(name_span.clone()),
            ExprOp::FnCall(0).with_span(MockSpan::merge(name_span.clone(), call_span)),
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

    #[test]
    fn diagnose_sizeof_of_symbol() {
        test_diagnosis_of_wrong_sizeof_arg(Atom::Name(NameId::Def(NameDefId(0))), ValueKind::Symbol)
    }

    #[test]
    fn diagnose_sizeof_of_sizeof() {
        test_diagnosis_of_wrong_sizeof_arg(
            Atom::Name(NameId::Builtin(BuiltinName::Sizeof)),
            ValueKind::Builtin,
        )
    }

    #[test]
    fn diagnose_sizeof_of_num() {
        test_diagnosis_of_wrong_sizeof_arg(Atom::Const(42), ValueKind::Num)
    }

    fn test_diagnosis_of_wrong_sizeof_arg(inner: Atom<LocationCounter, NameId>, found: ValueKind) {
        let program = &Program {
            sections: vec![],
            names: NameTable(vec![Some(NameDef::Symbol(Expr::from_atom(
                42.into(),
                MockSpan::from("42"),
            )))]),
            relocs: 0,
        };
        let relocs = &RelocTable::new(0);
        let context = EvalContext {
            program,
            relocs,
            location: Num::Unknown,
        };
        let mut diagnostics = MockDiagnostics::new(Log::new());
        let inner_span = MockSpan::from("inner");
        let sizeof_span = MockSpan::from("sizeof");
        let immediate = Expr(vec![
            ExprOp::Atom(inner).with_span(inner_span.clone()),
            ExprOp::Atom(Atom::Name(NameId::Builtin(BuiltinName::Sizeof)))
                .with_span(sizeof_span.clone()),
            ExprOp::FnCall(1).with_span(MockSpan::merge(sizeof_span, MockSpan::from("paren_r"))),
        ]);
        let num = immediate.to_num(&context, &mut diagnostics);
        assert_eq!(num, Num::Unknown);
        assert_eq!(
            diagnostics.into_log(),
            [DiagnosticsEvent::EmitDiag(
                Message::ExpectedFound {
                    expected: ValueKind::Section,
                    found,
                }
                .at(inner_span)
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

    fn eval_in_empty_program<S: Clone>(
        immediate: Immediate<S>,
        diagnostics: &mut impl BackendDiagnostics<S>,
    ) -> Num {
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
        immediate.to_num(context, diagnostics)
    }
}
