use super::{LinkageContext, VarTable};

use crate::diag::span::{Spanned, WithSpan};
use crate::diag::{BackendDiagnostics, Message, ValueKind};
use crate::model::{Atom, BinOp, Expr, ExprOp, LocationCounter, ParamId};
use crate::object::num::Num;
use crate::object::*;

use std::borrow::Borrow;

impl<S: Clone> Const<S> {
    pub(crate) fn to_num<'a, V, D>(
        &self,
        context: &LinkageContext<&'a Content<S>, V>,
        diagnostics: &mut D,
    ) -> Num
    where
        V: Borrow<VarTable>,
        D: BackendDiagnostics<S>,
    {
        self.eval(context, &[], diagnostics)
    }
}

trait Eval<'a, S: Clone> {
    type Output;

    fn eval<V: Borrow<VarTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a LinkageContext<&'a Content<S>, V>,
        args: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output;
}

#[derive(Clone)]
enum Value<'a, S: Clone> {
    Symbol(ResolvedSymbol<'a, S>),
    Num(Num),
    Unresolved,
}

#[derive(Clone)]
enum ResolvedSymbol<'a, S> {
    Section(&'a Section<S>),
    Sizeof,
    Expr(&'a Expr<Atom<VarId, Symbol>, S>),
}

impl<'a, L, S: Clone> Eval<'a, S> for &'a Expr<Atom<L, Symbol>, S>
where
    for<'r> Spanned<&'r Atom<L, Symbol>, &'r S>: Eval<'a, S, Output = Value<'a, S>>,
{
    type Output = Num;

    fn eval<V: Borrow<VarTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a LinkageContext<&'a Content<S>, V>,
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

    fn eval<V: Borrow<VarTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a LinkageContext<&'a Content<S>, V>,
        args: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output {
        match self.item {
            Value::Symbol(name) => name.with_span(self.span).eval(context, args, diagnostics),
            Value::Num(value) => value,
            Value::Unresolved => Num::Unknown,
        }
    }
}

impl<'a, S: Clone> Eval<'a, S> for Spanned<ResolvedSymbol<'a, S>, &S> {
    type Output = Num;

    fn eval<V: Borrow<VarTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a LinkageContext<&'a Content<S>, V>,
        args: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output {
        match self.item {
            ResolvedSymbol::Section(section) => context.vars.borrow()[section.addr].value.clone(),
            ResolvedSymbol::Sizeof => args
                .get(0)
                .map(|value| value.sizeof(context, diagnostics))
                .unwrap_or_else(|| {
                    let name = diagnostics.strip_span(self.span);
                    diagnostics.emit_diag(
                        Message::CannotCoerceBuiltinNameIntoNum { name }.at(self.span.clone()),
                    );
                    Num::Unknown
                }),
            ResolvedSymbol::Expr(expr) => expr.eval(context, args, diagnostics),
        }
    }
}

impl<'a, S: Clone + 'a> Eval<'a, S> for Spanned<&Atom<LocationCounter, Symbol>, &S> {
    type Output = Value<'a, S>;

    fn eval<V: Borrow<VarTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a LinkageContext<&'a Content<S>, V>,
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

impl<'a, S: Clone + 'a> Eval<'a, S> for Spanned<&Atom<VarId, Symbol>, &S> {
    type Output = Value<'a, S>;

    fn eval<V: Borrow<VarTable>, D: BackendDiagnostics<S>>(
        self,
        context: &'a LinkageContext<&'a Content<S>, V>,
        args: &'a [Spanned<Value<'a, S>, &S>],
        diagnostics: &mut D,
    ) -> Self::Output {
        match self.item {
            Atom::Const(value) => Value::Num((*value).into()),
            Atom::Location(id) => Value::Num(context.vars.borrow()[*id].value.clone()),
            Atom::Name(id) => (*id).with_span(self.span).to_value(context, diagnostics),
            Atom::Param(ParamId(id)) => args[*id].item.clone(),
        }
    }
}

impl<S: Clone> Spanned<Symbol, &S> {
    fn to_value<'a, V, D: BackendDiagnostics<S>>(
        &self,
        context: &'a LinkageContext<&'a Content<S>, V>,
        diagnostics: &mut D,
    ) -> Value<'a, S> {
        self.resolve(context, diagnostics)
            .map(Value::Symbol)
            .unwrap_or(Value::Unresolved)
    }

    fn resolve<'a, V, D: BackendDiagnostics<S>>(
        &self,
        context: &'a LinkageContext<&'a Content<S>, V>,
        diagnostics: &mut D,
    ) -> Option<ResolvedSymbol<'a, S>> {
        match self.item {
            Symbol::Builtin(BuiltinSymbol::Sizeof) => Some(ResolvedSymbol::Sizeof),
            Symbol::Content(id) => id.with_span(self.span).resolve(context, diagnostics),
        }
    }
}

impl<S: Clone> Spanned<ContentSymbol, &S> {
    fn resolve<'a, V, D: BackendDiagnostics<S>>(
        &self,
        context: &'a LinkageContext<&'a Content<S>, V>,
        diagnostics: &mut D,
    ) -> Option<ResolvedSymbol<'a, S>> {
        let id = self.item;
        let resolved = context.content.symbols.get(id).map(|def| match def {
            ContentDef::Section(SectionId(id)) => {
                ResolvedSymbol::Section(&context.content.sections[*id])
            }
            ContentDef::Expr(expr) => ResolvedSymbol::Expr(expr),
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
            BinOp::Division => lhs / rhs,
            BinOp::Minus => lhs - rhs,
            BinOp::Multiplication => lhs * rhs,
            BinOp::Plus => lhs + rhs,
        }
    }
}

impl<'a, S: Clone> Spanned<Value<'a, S>, &S> {
    fn sizeof<V, D>(
        &self,
        context: &'a LinkageContext<&'a Content<S>, V>,
        diagnostics: &mut D,
    ) -> Num
    where
        V: Borrow<VarTable>,
        D: BackendDiagnostics<S>,
    {
        match self.item {
            Value::Symbol(ResolvedSymbol::Section(section)) => {
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
            Value::Symbol(ResolvedSymbol::Section(_)) => Some(ValueKind::Section),
            Value::Symbol(ResolvedSymbol::Sizeof) => Some(ValueKind::Builtin),
            Value::Symbol(ResolvedSymbol::Expr(_)) => Some(ValueKind::Symbol),
            Value::Num(_) => Some(ValueKind::Num),
            Value::Unresolved => None,
        }
    }
}

pub const BUILTIN_SYMBOLS: &[(&str, Symbol)] =
    &[("sizeof", Symbol::Builtin(BuiltinSymbol::Sizeof))];

#[cfg(test)]
mod tests {
    use super::*;

    use crate::diag::*;
    use crate::log::Log;
    use crate::object::Var;

    type MockDiagnostics<S> = crate::diag::MockDiagnostics<DiagnosticsEvent<S>, S>;

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
            Const::from_atom(ContentSymbol(0).into(), ()).to_num(&context, &mut IgnoreDiagnostics),
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
            Const::from_items(&[
                ContentSymbol(0).into(),
                BuiltinSymbol::Sizeof.into(),
                ExprOp::FnCall(1).into()
            ])
            .to_num(context, &mut IgnoreDiagnostics),
            size.into()
        )
    }

    #[test]
    fn eval_fn_call_in_immediate() {
        let immediate =
            Const::from_items(&[42.into(), ContentSymbol(0).into(), ExprOp::FnCall(1).into()]);
        let content = &Content::<()> {
            sections: vec![],
            symbols: SymbolTable(vec![Some(ContentDef::Expr(Expr::from_items(&[
                ParamId(0).into(),
                1.into(),
                BinOp::Plus.into(),
            ])))]),
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
        let immediate = Const::from_items(&[ContentSymbol(0).into(), ExprOp::FnCall(0).into()]);
        assert_eq!(
            immediate.to_num(&context, &mut IgnoreDiagnostics),
            addr.into()
        )
    }

    #[test]
    fn eval_bitwise_or() {
        assert_eq!(
            eval_in_empty_program(
                Const::from_items(&[
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
    fn eval_division() {
        assert_eq!(
            eval_in_empty_program(
                Const::from_items(&[100.into(), 4.into(), ExprOp::Binary(BinOp::Division).into()]),
                &mut IgnoreDiagnostics
            ),
            25.into()
        )
    }

    #[test]
    fn diagnose_using_sizeof_as_immediate() {
        let mut diagnostics = MockDiagnostics::new(Log::new());
        let immediate = Const::from_atom(
            Atom::Name(Symbol::Builtin(BuiltinSymbol::Sizeof)),
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
        let mut diagnostics = MockDiagnostics::new(Log::new());
        let name_span = MockSpan::from("name");
        let call_span = MockSpan::from("call");
        let immediate = Expr(vec![
            ExprOp::Atom(Atom::Name(Symbol::Content(ContentSymbol(0))))
                .with_span(name_span.clone()),
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
        test_diagnosis_of_wrong_sizeof_arg(
            Atom::Name(Symbol::Content(ContentSymbol(0))),
            ValueKind::Symbol,
        )
    }

    #[test]
    fn diagnose_sizeof_of_sizeof() {
        test_diagnosis_of_wrong_sizeof_arg(
            Atom::Name(Symbol::Builtin(BuiltinSymbol::Sizeof)),
            ValueKind::Builtin,
        )
    }

    #[test]
    fn diagnose_sizeof_of_num() {
        test_diagnosis_of_wrong_sizeof_arg(Atom::Const(42), ValueKind::Num)
    }

    fn test_diagnosis_of_wrong_sizeof_arg(inner: Atom<LocationCounter, Symbol>, found: ValueKind) {
        let content = &Content {
            sections: vec![],
            symbols: SymbolTable(vec![Some(ContentDef::Expr(Expr::from_atom(
                42.into(),
                MockSpan::from("42"),
            )))]),
        };
        let vars = &VarTable(vec![]);
        let context = LinkageContext {
            content,
            vars,
            location: Num::Unknown,
        };
        let mut diagnostics = MockDiagnostics::new(Log::new());
        let inner_span = MockSpan::from("inner");
        let sizeof_span = MockSpan::from("sizeof");
        let immediate = Expr(vec![
            ExprOp::Atom(inner).with_span(inner_span.clone()),
            ExprOp::Atom(Atom::Name(Symbol::Builtin(BuiltinSymbol::Sizeof)))
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

    fn mk_program_with_empty_section<S>() -> Content<S> {
        Content {
            sections: vec![Section {
                constraints: Constraints { addr: None },
                addr: VarId(0),
                size: VarId(1),
                items: vec![],
            }],
            symbols: SymbolTable(vec![Some(ContentDef::Section(SectionId(0)))]),
        }
    }

    fn eval_in_empty_program<S: Clone>(
        immediate: Const<S>,
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
