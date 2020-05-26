use super::{BuiltinInstrSemantics, Core};

use crate::analyze::semantics::actions::Keyword;
use crate::analyze::semantics::arg::{Arg, DerefableArg};
use crate::analyze::semantics::reentrancy::Meta;
use crate::analyze::semantics::resolve::{NameTable, ResolvedName};
use crate::analyze::semantics::{ArgSemantics, ExprBuilder, Semantics};
use crate::analyze::syntax::actions::*;
use crate::analyze::Literal;
use crate::diag::span::Source;
use crate::diag::{Diagnostics, Message};
use crate::expr::{FnCall, LocationCounter, ParamId};
use crate::object::builder::{Finish, Name, PartialBackend, SymbolSource, ValueBuilder};

use std::ops::DerefMut;

delegate_diagnostics! {
    {R, S, P: Diagnostics<S>}, ExprBuilder<R, S, P>, {parent}, P, S
}

impl<'a, R, N, B> ArgFinalizer for ArgSemantics<'a, R, N, B>
where
    R: Meta,
    B: Finish,
    B::Value: Source<Span = R::Span>,
    B::Parent: PartialBackend<R::Span, Value = B::Value>,
{
    type Next = BuiltinInstrSemantics<'a, R, N, B::Parent>;

    fn did_parse_arg(mut self) -> Self::Next {
        let (builder, value) = self.core.builder.finish();
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
            core: Core {
                reentrancy: self.core.reentrancy,
                names: self.core.names,
                builder,
            },
            state: self.state.parent,
            tokens: self.tokens,
        }
    }
}

impl<'a, R, N, B> ArgContext for ArgSemantics<'a, R, N, B>
where
    R: Meta,
    N: DerefMut,
    N::Target: NameTable<
        R::Ident,
        Keyword = &'static Keyword,
        MacroId = R::MacroId,
        SymbolId = <<B as Finish>::Parent as SymbolSource>::SymbolId,
    >,
    B: ValueBuilder<<<B as Finish>::Parent as SymbolSource>::SymbolId, R::Span> + Finish,
    B::Parent: PartialBackend<R::Span>,
{
    fn act_on_atom(&mut self, atom: ExprAtom<R::Ident, Literal<R::StringRef>>, span: R::Span) {
        match atom {
            ExprAtom::Ident(ident) => self.act_on_ident(ident, span),
            ExprAtom::Literal(Literal::Number(n)) => {
                self.core.builder.push_op(n, span);
                self.state.arg = Some(Arg::Bare(DerefableArg::Const(())));
            }
            ExprAtom::Literal(Literal::String(string)) => {
                self.state.arg = Some(Arg::String(string, span))
            }
            ExprAtom::LocationCounter => {
                self.core.builder.push_op(LocationCounter, span);
                self.state.arg = Some(Arg::Bare(DerefableArg::Const(())));
            }
            ExprAtom::Error => self.state.arg = Some(Arg::Error),
        }
    }

    fn act_on_operator(&mut self, op: Operator, span: R::Span) {
        match op {
            Operator::Binary(op) => self.core.builder.push_op(op, span),
            Operator::FnCall(arity) => self.core.builder.push_op(FnCall(arity), span),
            Operator::Unary(UnaryOperator::Parentheses) => match &self.state.arg {
                Some(Arg::Bare(arg)) => self.state.arg = Some(Arg::Deref((*arg).clone(), span)),
                _ => unimplemented!(),
            },
        }
    }
}

impl<'a, R, N, B> ArgSemantics<'a, R, N, B>
where
    R: Meta,
    N: DerefMut,
    N::Target: NameTable<
        R::Ident,
        Keyword = &'static Keyword,
        MacroId = R::MacroId,
        SymbolId = <<B as Finish>::Parent as SymbolSource>::SymbolId,
    >,
    B: ValueBuilder<<<B as Finish>::Parent as SymbolSource>::SymbolId, R::Span> + Finish,
    B::Parent: PartialBackend<R::Span>,
{
    fn act_on_ident(&mut self, ident: R::Ident, span: R::Span) {
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
            self.core.builder.push_op(id, span);
            self.state.arg = Some(Arg::Bare(DerefableArg::Const(())));
            return;
        }
        match self.core.names.resolve_name(&ident) {
            Some(ResolvedName::Keyword(Keyword::Operand(symbol))) => match self.state.arg {
                None => self.state.arg = Some(Arg::Bare(DerefableArg::Symbol(*symbol, span))),
                _ => unimplemented!(),
            },
            Some(ResolvedName::Keyword(_)) => {
                let keyword = self.strip_span(&span);
                self.emit_diag(Message::KeywordInExpr { keyword }.at(span))
            }
            Some(ResolvedName::Symbol(id)) => {
                self.core.builder.push_op(Name(id), span);
                self.state.arg = Some(Arg::Bare(DerefableArg::Const(())))
            }
            None => {
                let id = self.core.builder.alloc_symbol(span.clone());
                self.define_name(ident, ResolvedName::Symbol(id.clone()));
                self.core.builder.push_op(Name(id), span);
                self.state.arg = Some(Arg::Bare(DerefableArg::Const(())))
            }
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::analyze::semantics::actions::tests::collect_semantic_actions;
    use crate::analyze::semantics::resolve::{NameTableEvent, ResolvedName};
    use crate::analyze::syntax::actions::*;
    use crate::diag::span::WithSpan;
    use crate::diag::{DiagnosticsEvent, Message, MockSpan};
    use crate::expr::{Atom, BinOp, Expr, ExprOp};
    use crate::object::builder::mock::{BackendEvent, MockSymbolId};
    use crate::object::builder::{CpuInstr, Direction, Item, Ld, SpecialLd, Width};

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
}
