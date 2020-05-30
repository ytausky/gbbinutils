use super::BuiltinInstrSemantics;

use crate::analyze::macros::MacroSource;
use crate::analyze::{Literal, StringSource};
use crate::diag::span::{Source, SpanSource};
use crate::diag::{Diagnostics, Message};
use crate::expr::{FnCall, LocationCounter, ParamId};
use crate::semantics::actions::Keyword;
use crate::semantics::arg::{Arg, DerefableArg};
use crate::semantics::{ArgSemantics, Semantics};
use crate::session::builder::{Finish, Name, SymbolSource, ValueBuilder};
use crate::session::resolve::{NameTable, ResolvedName};
use crate::session::Session;
use crate::syntax::actions::*;
use crate::syntax::IdentSource;

impl<'a, S> ArgFinalizer for ArgSemantics<'a, S>
where
    S: Finish,
    S::Value: Source<Span = <S::Parent as SpanSource>::Span>,
    S::Parent: Session<Value = S::Value>,
{
    type Next = BuiltinInstrSemantics<'a, S::Parent>;

    fn did_parse_arg(mut self) -> Self::Next {
        let (session, value) = self.session.finish();
        let arg = match self.state.arg {
            Some(Arg::Bare(DerefableArg::Const(()))) => {
                Arg::Bare(DerefableArg::Const(value.unwrap()))
            }
            Some(Arg::Bare(DerefableArg::Symbol(symbol, span))) => {
                Arg::Bare(DerefableArg::Symbol(symbol, span))
            }
            Some(Arg::Deref(DerefableArg::Const(()), span)) => {
                Arg::Deref(DerefableArg::Const(value.unwrap()), span)
            }
            Some(Arg::Deref(DerefableArg::Symbol(symbol, inner_span), outer_span)) => {
                Arg::Deref(DerefableArg::Symbol(symbol, inner_span), outer_span)
            }
            Some(Arg::String(string, span)) => Arg::String(string, span),
            Some(Arg::Error) | None => Arg::Error,
        };
        self.state.parent.args.push(arg);
        Semantics {
            session,
            state: self.state.parent,
            tokens: self.tokens,
        }
    }
}

impl<'a, S> ArgContext for ArgSemantics<'a, S>
where
    S: NameTable<
        <<S as Finish>::Parent as IdentSource>::Ident,
        Keyword = &'static Keyword,
        MacroId = <<S as Finish>::Parent as MacroSource>::MacroId,
        SymbolId = <<S as Finish>::Parent as SymbolSource>::SymbolId,
    >,
    S: ValueBuilder<
            <<S as Finish>::Parent as SpanSource>::Span,
            SymbolId = <<S as Finish>::Parent as SymbolSource>::SymbolId,
        > + Finish,
    S: Diagnostics<<<S as Finish>::Parent as SpanSource>::Span>,
    S::Parent: Session,
{
    fn act_on_atom(
        &mut self,
        atom: ExprAtom<
            <<S as Finish>::Parent as IdentSource>::Ident,
            Literal<<<S as Finish>::Parent as StringSource>::StringRef>,
        >,
        span: <<S as Finish>::Parent as SpanSource>::Span,
    ) {
        match atom {
            ExprAtom::Ident(ident) => self.act_on_ident(ident, span),
            ExprAtom::Literal(Literal::Number(n)) => {
                self.session.push_op(n, span);
                self.state.arg = Some(Arg::Bare(DerefableArg::Const(())));
            }
            ExprAtom::Literal(Literal::String(string)) => {
                self.state.arg = Some(Arg::String(string, span))
            }
            ExprAtom::LocationCounter => {
                self.session.push_op(LocationCounter, span);
                self.state.arg = Some(Arg::Bare(DerefableArg::Const(())));
            }
            ExprAtom::Error => self.state.arg = Some(Arg::Error),
        }
    }

    fn act_on_operator(&mut self, op: Operator, span: <<S as Finish>::Parent as SpanSource>::Span) {
        match op {
            Operator::Binary(op) => self.session.push_op(op, span),
            Operator::FnCall(arity) => self.session.push_op(FnCall(arity), span),
            Operator::Unary(UnaryOperator::Parentheses) => match &self.state.arg {
                Some(Arg::Bare(arg)) => self.state.arg = Some(Arg::Deref((*arg).clone(), span)),
                _ => unimplemented!(),
            },
        }
    }
}

impl<'a, S> ArgSemantics<'a, S>
where
    S: NameTable<
        <<S as Finish>::Parent as IdentSource>::Ident,
        Keyword = &'static Keyword,
        MacroId = <<S as Finish>::Parent as MacroSource>::MacroId,
        SymbolId = <<S as Finish>::Parent as SymbolSource>::SymbolId,
    >,
    S: ValueBuilder<
            <<S as Finish>::Parent as SpanSource>::Span,
            SymbolId = <<S as Finish>::Parent as SymbolSource>::SymbolId,
        > + Finish,
    S: Diagnostics<<<S as Finish>::Parent as SpanSource>::Span>,
    S::Parent: Session,
{
    fn act_on_ident(
        &mut self,
        ident: <<S as Finish>::Parent as IdentSource>::Ident,
        span: <<S as Finish>::Parent as SpanSource>::Span,
    ) {
        let no_params = (vec![], vec![]);
        let params = match &self.state.parent.label {
            Some((_, params)) => &params,
            _ => &no_params,
        };
        let param = params
            .0
            .iter()
            .position(|param| *param == ident)
            .map(ParamId);
        if let Some(id) = param {
            self.session.push_op(id, span);
            self.state.arg = Some(Arg::Bare(DerefableArg::Const(())));
            return;
        }
        match self.session.resolve_name(&ident) {
            Some(ResolvedName::Keyword(Keyword::Operand(symbol))) => match self.state.arg {
                None => self.state.arg = Some(Arg::Bare(DerefableArg::Symbol(*symbol, span))),
                _ => unimplemented!(),
            },
            Some(ResolvedName::Keyword(_)) => {
                let keyword = self.session.strip_span(&span);
                self.session
                    .emit_diag(Message::KeywordInExpr { keyword }.at(span))
            }
            Some(ResolvedName::Symbol(id)) => {
                self.session.push_op(Name(id), span);
                self.state.arg = Some(Arg::Bare(DerefableArg::Const(())))
            }
            None => {
                let id = self.session.alloc_symbol(span.clone());
                self.session
                    .define_name(ident, ResolvedName::Symbol(id.clone()));
                self.session.push_op(Name(id), span);
                self.state.arg = Some(Arg::Bare(DerefableArg::Const(())))
            }
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::diag::span::WithSpan;
    use crate::diag::{DiagnosticsEvent, Message, MockSpan};
    use crate::expr::{Atom, BinOp, Expr, ExprOp, ParamId};
    use crate::semantics::actions::tests::collect_semantic_actions;
    use crate::session::builder::mock::{BackendEvent, MockSymbolId};
    use crate::session::builder::{CpuInstr, Direction, Item, Ld, SpecialLd, Width};
    use crate::session::resolve::{NameTableEvent, ResolvedName};
    use crate::syntax::actions::*;

    #[test]
    fn diagnose_keyword_in_expr() {
        assert_eq!(
            collect_semantic_actions::<_, MockSpan<_>>(|actions| {
                let mut actions = actions
                    .will_parse_line()
                    .into_instr_line()
                    .will_parse_instr("DB".into(), "db".into())
                    .into_builtin_instr()
                    .will_parse_arg();
                actions.act_on_atom(ExprAtom::Ident("DB".into()), "keyword".into());
                actions
                    .did_parse_arg()
                    .did_parse_instr()
                    .did_parse_line("eol".into())
                    .act_on_eos("eos".into())
            }),
            [DiagnosticsEvent::EmitDiag(
                Message::KeywordInExpr {
                    keyword: "keyword".into()
                }
                .at("keyword".into())
                .into()
            )
            .into()]
        )
    }

    #[test]
    fn handle_unknown_name() {
        let actual = collect_semantic_actions::<_, MockSpan<_>>(|actions| {
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("DB".into(), "db".into())
                .into_builtin_instr()
                .will_parse_arg();
            actions.act_on_atom(ExprAtom::Ident("f".into()), "f".into());
            actions
                .did_parse_arg()
                .did_parse_instr()
                .did_parse_line("eol".into())
                .act_on_eos("eos".into())
        });
        let expected = [
            NameTableEvent::Insert("f".into(), ResolvedName::Symbol(MockSymbolId(0))).into(),
            BackendEvent::EmitItem(Item::Data(
                Expr::from_atom(Atom::Name(MockSymbolId(0)), "f".into()),
                Width::Byte,
            ))
            .into(),
        ];
        assert_eq!(actual, expected)
    }

    #[test]
    fn act_on_known_symbol_name() {
        let actual = collect_semantic_actions::<_, MockSpan<_>>(|actions| {
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("DB".into(), "db".into())
                .into_builtin_instr()
                .will_parse_arg();
            actions.act_on_atom(ExprAtom::Ident("f".into()), "f1".into());
            actions.act_on_atom(ExprAtom::Ident("f".into()), "f2".into());
            actions.act_on_operator(Operator::Binary(BinOp::Plus), "plus".into());
            actions
                .did_parse_arg()
                .did_parse_instr()
                .did_parse_line("eol".into())
                .act_on_eos("eos".into())
        });
        let expected = [
            NameTableEvent::Insert("f".into(), ResolvedName::Symbol(MockSymbolId(0))).into(),
            BackendEvent::EmitItem(Item::Data(
                Expr(vec![
                    ExprOp::Atom(Atom::Name(MockSymbolId(0))).with_span("f1".into()),
                    ExprOp::Atom(Atom::Name(MockSymbolId(0))).with_span("f2".into()),
                    ExprOp::Binary(BinOp::Plus).with_span("plus".into()),
                ]),
                Width::Byte,
            ))
            .into(),
        ];
        assert_eq!(actual, expected)
    }

    #[test]
    fn handle_deref_const() {
        let actual = collect_semantic_actions::<_, MockSpan<_>>(|actions| {
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("LD".into(), "ld".into())
                .into_builtin_instr()
                .will_parse_arg();
            actions.act_on_atom(ExprAtom::Ident("A".into()), "a".into());
            let mut actions = actions.did_parse_arg().will_parse_arg();
            actions.act_on_atom(ExprAtom::Ident("const".into()), "const".into());
            actions.act_on_operator(Operator::Unary(UnaryOperator::Parentheses), "deref".into());
            actions
                .did_parse_arg()
                .did_parse_instr()
                .did_parse_line("eol".into())
                .act_on_eos("eos".into())
        });
        let expected = [
            NameTableEvent::Insert("const".into(), ResolvedName::Symbol(MockSymbolId(0))).into(),
            BackendEvent::EmitItem(Item::CpuInstr(CpuInstr::Ld(Ld::Special(
                SpecialLd::InlineAddr(Expr::from_atom(Atom::Name(MockSymbolId(0)), "const".into())),
                Direction::IntoA,
            ))))
            .into(),
        ];
        assert_eq!(actual, expected)
    }

    #[test]
    fn handle_param() {
        let actual = collect_semantic_actions::<_, MockSpan<_>>(|context| {
            let mut context = context
                .will_parse_line()
                .into_instr_line()
                .will_parse_label(("label".into(), "label".into()));
            context.act_on_param("param".into(), "param1".into());
            let mut context = context
                .did_parse_label()
                .will_parse_instr("EQU".into(), "equ".into())
                .into_builtin_instr()
                .will_parse_arg();
            context.act_on_atom(ExprAtom::Ident("param".into()), "param2".into());
            context
                .did_parse_arg()
                .did_parse_instr()
                .did_parse_line("eol".into())
        });
        let expected = [
            NameTableEvent::Insert("label".into(), ResolvedName::Symbol(MockSymbolId(0))).into(),
            BackendEvent::DefineSymbol(
                (MockSymbolId(0), "label".into()),
                Expr::from_atom(Atom::Param(ParamId(0)), "param2".into()),
            )
            .into(),
        ];
        assert_eq!(actual, expected)
    }
}
