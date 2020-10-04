use super::{Expr, LinkageContext, VarTable};

use crate::expr::{Atom, BinOp, ExprOp, ParamId};
use crate::object::num::Num;
use crate::object::*;
use crate::session::diagnostics::{BackendDiagnostics, Message, ValueKind};
use crate::span::{Spanned, WithSpan};

use std::borrow::Borrow;

impl<S: Clone> Expr<S> {
    pub(crate) fn to_num<C, V, D>(&self, context: &LinkageContext<C, V>, diagnostics: &mut D) -> Num
    where
        C: Borrow<Content<S>>,
        V: Borrow<VarTable>,
        D: BackendDiagnostics<S>,
    {
        self.eval_subst(
            &EvalContext {
                linkage: context,
                args: &[],
                location_var: None,
            },
            diagnostics,
        )
    }
}

trait EvalSubst<'a, S: Clone> {
    type Output;

    fn eval_subst<C: Borrow<Content<S>>, V: Borrow<VarTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a EvalContext<'a, C, V, S>,
        diagnostics: &mut D,
    ) -> Self::Output;
}

struct EvalContext<'a, C, V, S: Clone> {
    linkage: &'a LinkageContext<C, V>,
    args: Args<'a, S>,
    location_var: Option<VarId>,
}

impl<'a, C, V: Borrow<VarTable>, S: Clone> EvalContext<'a, C, V, S> {
    fn location(&self) -> Num {
        match self.location_var {
            Some(id) => self.linkage.vars.borrow()[id].value.clone(),
            None => self.linkage.location.clone(),
        }
    }
}

#[derive(Clone)]
enum Value<'a, S: Clone> {
    Symbol(Option<DefRef<'a, S>>),
    Num(Num),
}

type DefRef<'a, S> = Symbol<BuiltinDefId, UserDef<&'a Closure<S>, &'a Section<S>>>;

impl<'a, S: Clone> EvalSubst<'a, S> for &'a Expr<S> {
    type Output = Num;

    fn eval_subst<C: Borrow<Content<S>>, V: Borrow<VarTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a EvalContext<'a, C, V, S>,
        diagnostics: &mut D,
    ) -> Self::Output {
        let mut stack = Vec::<Spanned<Value<_>, _>>::new();
        for Spanned { item, span } in &self.0 {
            let value = match item {
                ExprOp::Atom(atom) => atom.with_span(span).eval_subst(context, diagnostics),
                ExprOp::Binary(operator) => {
                    let rhs = stack.pop().unwrap();
                    let lhs = stack.pop().unwrap().eval_subst(context, diagnostics);
                    let rhs = rhs.eval_subst(context, diagnostics);
                    Value::Num(operator.apply(&lhs, &rhs))
                }
                ExprOp::FnCall(n) => {
                    let arg_index = stack.len() - n;
                    let name = stack[arg_index - 1].clone();
                    let value = Value::Num(name.eval_subst(
                        &EvalContext {
                            linkage: context.linkage,
                            args: &stack[arg_index..],
                            location_var: context.location_var,
                        },
                        diagnostics,
                    ));
                    stack.truncate(arg_index - 1);
                    value
                }
            };
            stack.push(value.with_span(span))
        }
        stack.pop().unwrap().eval_subst(context, diagnostics)
    }
}

type Args<'a, S> = &'a [Spanned<Value<'a, S>, &'a S>];

impl<'a, S: Clone> EvalSubst<'a, S> for Spanned<Value<'a, S>, &S> {
    type Output = Num;

    fn eval_subst<C: Borrow<Content<S>>, V: Borrow<VarTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a EvalContext<'a, C, V, S>,
        diagnostics: &mut D,
    ) -> Self::Output {
        match self.item {
            Value::Symbol(Some(name)) => name.with_span(self.span).eval_subst(context, diagnostics),
            Value::Symbol(None) => Num::Unknown,
            Value::Num(value) => value,
        }
    }
}

impl<'a, S: Clone> EvalSubst<'a, S> for Spanned<DefRef<'a, S>, &S> {
    type Output = Num;

    fn eval_subst<C: Borrow<Content<S>>, V: Borrow<VarTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a EvalContext<'a, C, V, S>,
        diagnostics: &mut D,
    ) -> Self::Output {
        match self.item {
            Symbol::Builtin(BuiltinDefId::Sizeof) => context
                .args
                .get(0)
                .map(|value| value.sizeof(context.linkage, diagnostics))
                .unwrap_or_else(|| {
                    let name = diagnostics.strip_span(self.span);
                    diagnostics.emit_diag(
                        Message::CannotCoerceBuiltinNameIntoNum { name }.at(self.span.clone()),
                    );
                    Num::Unknown
                }),
            Symbol::UserDef(UserDef::Closure(def)) => def.expr.eval_subst(
                &EvalContext {
                    location_var: Some(def.location),
                    ..*context
                },
                diagnostics,
            ),
            Symbol::UserDef(UserDef::Section(section)) => {
                context.linkage.vars.borrow()[section.addr].value.clone()
            }
        }
    }
}

impl<'a, S: Clone + 'a> EvalSubst<'a, S> for Spanned<&Atom<SymbolId>, &S> {
    type Output = Value<'a, S>;

    fn eval_subst<C: Borrow<Content<S>>, V: Borrow<VarTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a EvalContext<'a, C, V, S>,
        diagnostics: &mut D,
    ) -> Self::Output {
        match self.item {
            Atom::Const(value) => Value::Num((*value).into()),
            Atom::Location => Value::Num(context.location()),
            Atom::Name(id) => (*id)
                .with_span(self.span)
                .to_value(context.linkage, diagnostics),
            Atom::Param(ParamId(id)) => context.args[*id].item.clone(),
        }
    }
}

impl<S: Clone> Spanned<SymbolId, &S> {
    fn to_value<'a, C: Borrow<Content<S>>, V, D: BackendDiagnostics<S>>(
        &self,
        context: &'a LinkageContext<C, V>,
        diagnostics: &mut D,
    ) -> Value<'a, S> {
        Value::Symbol(self.resolve(context, diagnostics))
    }

    fn resolve<'a, C: Borrow<Content<S>>, V, D: BackendDiagnostics<S>>(
        &self,
        context: &'a LinkageContext<C, V>,
        diagnostics: &mut D,
    ) -> Option<DefRef<'a, S>> {
        match self.item {
            Symbol::Builtin(BuiltinDefId::Sizeof) => Some(Symbol::Builtin(BuiltinDefId::Sizeof)),
            Symbol::UserDef(id) => id.with_span(self.span).resolve(context, diagnostics),
        }
    }
}

impl<S: Clone> Spanned<UserDefId, &S> {
    fn resolve<'a, C: Borrow<Content<S>>, V, D: BackendDiagnostics<S>>(
        &self,
        context: &'a LinkageContext<C, V>,
        diagnostics: &mut D,
    ) -> Option<DefRef<'a, S>> {
        let id = self.item;
        let resolved = context
            .content
            .borrow()
            .symbols
            .get(id)
            .map(|def| match def {
                UserDef::Closure(formula) => UserDef::Closure(formula),
                UserDef::Section(SectionId(id)) => {
                    UserDef::Section(&context.content.borrow().sections[*id])
                }
            });
        if resolved.is_none() {
            let symbol = diagnostics.strip_span(self.span);
            diagnostics.emit_diag(Message::UnresolvedSymbol { symbol }.at(self.span.clone()))
        }
        resolved.map(Symbol::UserDef)
    }
}

impl BinOp {
    fn apply(self, lhs: &Num, rhs: &Num) -> Num {
        match self {
            BinOp::BitOr => lhs | rhs,
            BinOp::Division => lhs / rhs,
            BinOp::Equality => match (lhs.exact(), rhs.exact()) {
                (Some(lhs), Some(rhs)) => (if lhs == rhs { 1 } else { 0 }).into(),
                _ => Num::Unknown,
            },
            BinOp::Minus => lhs - rhs,
            BinOp::Multiplication => lhs * rhs,
            BinOp::Plus => lhs + rhs,
        }
    }
}

impl<'a, S: Clone> Spanned<Value<'a, S>, &S> {
    fn sizeof<C, V, D>(&self, context: &'a LinkageContext<C, V>, diagnostics: &mut D) -> Num
    where
        C: Borrow<Content<S>>,
        V: Borrow<VarTable>,
        D: BackendDiagnostics<S>,
    {
        match self.item {
            Value::Symbol(Some(Symbol::UserDef(UserDef::Section(section)))) => {
                context.vars.borrow()[section.size].value.clone()
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
            Value::Symbol(Some(Symbol::Builtin(_))) => Some(ValueKind::Builtin),
            Value::Symbol(Some(Symbol::UserDef(UserDef::Closure(_)))) => Some(ValueKind::Symbol),
            Value::Symbol(Some(Symbol::UserDef(UserDef::Section(_)))) => Some(ValueKind::Section),
            Value::Symbol(None) => None,
            Value::Num(_) => Some(ValueKind::Num),
        }
    }
}

pub const BUILTIN_SYMBOLS: &[(&str, SymbolId)] =
    &[("sizeof", Symbol::Builtin(BuiltinDefId::Sizeof))];

#[cfg(test)]
mod tests {
    use super::*;

    use crate::log::Log;
    use crate::object::Var;
    use crate::session::diagnostics::*;

    type MockDiagnostics<S> = crate::session::diagnostics::MockDiagnostics<DiagnosticsEvent<S>, S>;

    #[test]
    fn eval_section_addr() {
        let addr = 0x0100;
        let content = &mk_program_with_empty_section();
        let vars = &VarTable(vec![Var { value: addr.into() }, Var { value: 0.into() }]);
        let context = LinkageContext {
            content,
            vars,
            location: Num::Unknown,
        };
        assert_eq!(
            Expr::from_atom(UserDefId(0).into(), ()).to_num(&context, &mut IgnoreDiagnostics),
            addr.into()
        )
    }

    #[test]
    fn eval_section_size() {
        let content = &mk_program_with_empty_section();
        let size = 42;
        let vars = &VarTable(vec![Var { value: 0.into() }, Var { value: size.into() }]);
        let context = &LinkageContext {
            content,
            vars,
            location: Num::Unknown,
        };
        assert_eq!(
            Expr::from_items(&[
                BuiltinDefId::Sizeof.into(),
                UserDefId(0).into(),
                ExprOp::FnCall(1).into()
            ])
            .to_num(context, &mut IgnoreDiagnostics),
            size.into()
        )
    }

    #[test]
    fn eval_fn_call_in_immediate() {
        let immediate =
            Expr::from_items(&[UserDefId(0).into(), 42.into(), ExprOp::FnCall(1).into()]);
        let content = &Content::<()> {
            sections: vec![],
            symbols: SymbolTable(vec![Some(UserDef::Closure(Closure {
                expr: Expr::from_items(&[ParamId(0).into(), 1.into(), BinOp::Plus.into()]),
                location: VarId(0),
            }))]),
        };
        let vars = &VarTable(Vec::new());
        let context = &LinkageContext {
            content,
            vars,
            location: Num::Unknown,
        };
        assert_eq!(immediate.to_num(context, &mut IgnoreDiagnostics), 43.into())
    }

    #[test]
    fn eval_complex_expression() {
        let immediate = Expr::from_items(&[
            0.into(),
            UserDefId(0).into(),
            42.into(),
            ExprOp::FnCall(1).into(),
            BinOp::Plus.into(),
        ]);
        let content = &Content::<()> {
            sections: vec![],
            symbols: SymbolTable(vec![Some(UserDef::Closure(Closure {
                expr: Expr::from_items(&[ParamId(0).into(), 1.into(), BinOp::Plus.into()]),
                location: VarId(0),
            }))]),
        };
        let vars = &VarTable(Vec::new());
        let context = &LinkageContext {
            content,
            vars,
            location: Num::Unknown,
        };
        assert_eq!(immediate.to_num(context, &mut IgnoreDiagnostics), 43.into())
    }

    #[test]
    fn eval_section_name_call() {
        let addr = 0x1337;
        let content = &mk_program_with_empty_section();
        let vars = &VarTable(vec![Var { value: addr.into() }, Var { value: 0.into() }]);
        let context = LinkageContext {
            content,
            vars,
            location: Num::Unknown,
        };
        let immediate = Expr::from_items(&[UserDefId(0).into(), ExprOp::FnCall(0).into()]);
        assert_eq!(
            immediate.to_num(&context, &mut IgnoreDiagnostics),
            addr.into()
        )
    }

    #[test]
    fn eval_bitwise_or() {
        assert_eq!(
            eval_in_empty_program(
                Expr::from_items(&[
                    0x17.into(),
                    0x86.into(),
                    ExprOp::Binary(BinOp::BitOr).into(),
                ]),
                &mut IgnoreDiagnostics
            ),
            0x97.into()
        )
    }

    #[test]
    fn eval_division() {
        assert_eq!(
            eval_in_empty_program(
                Expr::from_items(&[100.into(), 4.into(), ExprOp::Binary(BinOp::Division).into()]),
                &mut IgnoreDiagnostics
            ),
            25.into()
        )
    }

    #[test]
    fn eval_known_false_equality() {
        assert_eq!(
            eval_in_empty_program(
                Expr::from_items(&[12.into(), 34.into(), ExprOp::Binary(BinOp::Equality).into()]),
                &mut IgnoreDiagnostics
            ),
            0.into()
        )
    }

    #[test]
    fn eval_known_true_equality() {
        assert_eq!(
            eval_in_empty_program(
                Expr::from_items(&[42.into(), 42.into(), ExprOp::Binary(BinOp::Equality).into()]),
                &mut IgnoreDiagnostics
            ),
            1.into()
        )
    }

    #[test]
    fn diagnose_using_sizeof_as_immediate() {
        let log = Log::new();
        let registry = &mut TestDiagnosticsListener::new();
        let mut diagnostics = MockDiagnostics::new(log.clone());
        let mut view = DiagnosticsView {
            codebase: &mut (),
            registry,
            diagnostics: &mut diagnostics,
        };
        let immediate = Expr::from_atom(
            Atom::Name(Symbol::Builtin(BuiltinDefId::Sizeof)),
            MockSpan::from(0),
        );
        let value = eval_in_empty_program(immediate, &mut view);
        drop(diagnostics);
        assert_eq!(value, Num::Unknown);
        assert_eq!(
            log.into_inner(),
            [DiagnosticsEvent::EmitDiag(
                Message::CannotCoerceBuiltinNameIntoNum { name: 0.into() }
                    .at(0.into())
                    .into()
            )]
        )
    }

    #[test]
    fn diagnose_calling_undefined_symbol() {
        let content = &Content {
            sections: vec![],
            symbols: SymbolTable(vec![None]),
        };
        let vars = &VarTable(vec![]);
        let context = LinkageContext {
            content,
            vars,
            location: Num::Unknown,
        };
        let log = Log::new();
        let registry = &mut TestDiagnosticsListener::new();
        let mut diagnostics = MockDiagnostics::new(log.clone());
        let mut view = DiagnosticsView {
            codebase: &mut (),
            registry,
            diagnostics: &mut diagnostics,
        };
        let name_span = MockSpan::from("name");
        let call_span = MockSpan::from("call");
        let immediate = crate::expr::Expr(vec![
            ExprOp::Atom(Atom::Name(Symbol::UserDef(UserDefId(0)))).with_span(name_span.clone()),
            ExprOp::FnCall(0).with_span(MockSpan::merge(name_span.clone(), call_span)),
        ]);
        let value = immediate.to_num(&context, &mut view);
        drop(diagnostics);
        assert_eq!(value, Num::Unknown);
        assert_eq!(
            log.into_inner(),
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
        test_diagnosis_of_wrong_sizeof_arg(
            Atom::Name(Symbol::UserDef(UserDefId(0))),
            ValueKind::Symbol,
        )
    }

    #[test]
    fn diagnose_sizeof_of_sizeof() {
        test_diagnosis_of_wrong_sizeof_arg(
            Atom::Name(Symbol::Builtin(BuiltinDefId::Sizeof)),
            ValueKind::Builtin,
        )
    }

    #[test]
    fn diagnose_sizeof_of_num() {
        test_diagnosis_of_wrong_sizeof_arg(Atom::Const(42), ValueKind::Num)
    }

    fn test_diagnosis_of_wrong_sizeof_arg(inner: Atom<SymbolId>, found: ValueKind) {
        let content = &Content {
            sections: vec![],
            symbols: SymbolTable(vec![Some(UserDef::Closure(Closure {
                expr: Expr::from_atom(42.into(), MockSpan::from("42")),
                location: VarId(0),
            }))]),
        };
        let vars = &VarTable(vec![]);
        let context = LinkageContext {
            content,
            vars,
            location: Num::Unknown,
        };
        let log = Log::new();
        let registry = &mut TestDiagnosticsListener::new();
        let mut diagnostics = MockDiagnostics::new(log.clone());
        let mut view = DiagnosticsView {
            codebase: &mut (),
            registry,
            diagnostics: &mut diagnostics,
        };
        let inner_span = MockSpan::from("inner");
        let sizeof_span = MockSpan::from("sizeof");
        let immediate = crate::expr::Expr(vec![
            ExprOp::Atom(Atom::Name(Symbol::Builtin(BuiltinDefId::Sizeof)))
                .with_span(sizeof_span.clone()),
            ExprOp::Atom(inner).with_span(inner_span.clone()),
            ExprOp::FnCall(1).with_span(MockSpan::merge(sizeof_span, MockSpan::from("paren_r"))),
        ]);
        let num = immediate.to_num(&context, &mut view);
        drop(diagnostics);
        assert_eq!(num, Num::Unknown);
        assert_eq!(
            log.into_inner(),
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

    fn mk_program_with_empty_section<S>() -> Content<S> {
        Content {
            sections: vec![Section {
                constraints: Constraints { addr: None },
                addr: VarId(0),
                size: VarId(1),
                fragments: vec![],
            }],
            symbols: SymbolTable(vec![Some(UserDef::Section(SectionId(0)))]),
        }
    }

    fn eval_in_empty_program<S: Clone>(
        immediate: Expr<S>,
        diagnostics: &mut impl BackendDiagnostics<S>,
    ) -> Num {
        let content = &Content {
            sections: vec![],
            symbols: SymbolTable(vec![]),
        };
        let vars = &VarTable(vec![]);
        let context = &LinkageContext {
            content,
            vars,
            location: Num::Unknown,
        };
        immediate.to_num(context, diagnostics)
    }
}
