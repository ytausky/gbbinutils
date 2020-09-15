use super::BuiltinInstrSemantics;

use crate::analyze::Literal;
use crate::diag::span::WithSpan;
use crate::expr::{Atom, Expr, ExprOp, ParamId};
use crate::semantics::arg::ParsedArg;
use crate::semantics::{ArgSemantics, Semantics};
use crate::session::Session;
use crate::syntax::actions::*;

impl<'a, 'b, S: Session> ArgFinalizer for ArgSemantics<'a, 'b, S> {
    type Next = BuiltinInstrSemantics<'a, 'b, S>;

    fn did_parse_arg(mut self) -> Self::Next {
        let arg = self.state.arg.unwrap_or(ParsedArg::Error);
        self.state.parent.args.push(arg);
        Semantics {
            session: self.session,
            state: self.state.parent,
            tokens: self.tokens,
        }
    }
}

impl<'a, 'b, S: Session> ArgContext for ArgSemantics<'a, 'b, S> {
    fn act_on_atom(&mut self, atom: ExprAtom<S::Ident, Literal<S::StringRef>>, span: S::Span) {
        match atom {
            ExprAtom::Ident(ident) => self.act_on_ident(ident, span),
            ExprAtom::Literal(Literal::Number(n)) => {
                self.act_on_expr_node(ExprOp::Atom(Atom::Const(n)), span)
            }
            ExprAtom::Literal(Literal::String(string)) => {
                self.state.arg = Some(ParsedArg::String(string, span))
            }
            ExprAtom::LocationCounter => self.act_on_expr_node(ExprOp::Atom(Atom::Location), span),
            ExprAtom::Error => self.state.arg = Some(ParsedArg::Error),
        }
    }

    fn act_on_operator(&mut self, op: Operator, span: S::Span) {
        match op {
            Operator::Binary(op) => self.act_on_expr_node(ExprOp::Binary(op), span),
            Operator::FnCall(arity) => self.act_on_expr_node(ExprOp::FnCall(arity), span),
            Operator::Unary(UnaryOperator::Parentheses) => match &self.state.arg {
                Some(ParsedArg::Bare(arg)) => {
                    self.state.arg = Some(ParsedArg::Parenthesized((*arg).clone(), span))
                }
                _ => unimplemented!(),
            },
        }
    }
}

impl<'a, 'b, S: Session> ArgSemantics<'a, 'b, S> {
    fn act_on_expr_node(&mut self, node: ExprOp<S::Ident>, span: S::Span) {
        self.state.arg = match self.state.arg.take() {
            None => Some(ParsedArg::Bare(Expr(vec![node.with_span(span)]))),
            Some(ParsedArg::Bare(mut expr)) | Some(ParsedArg::Parenthesized(mut expr, _)) => {
                expr.0.push(node.with_span(span));
                Some(ParsedArg::Bare(expr))
            }
            Some(ParsedArg::Error) => Some(ParsedArg::Error),
            Some(ParsedArg::String(_, _)) => todo!(),
        }
    }

    fn act_on_ident(&mut self, ident: S::Ident, span: S::Span) {
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
        match param {
            None => self.act_on_expr_node(ExprOp::Atom(Atom::Name(ident)), span),
            Some(id) => self.act_on_expr_node(ExprOp::Atom(Atom::Param(id)), span),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::diag::span::WithSpan;
    use crate::diag::{DiagnosticsEvent, Message, MockSpan};
    use crate::expr::{Atom, BinOp, Expr, ExprOp, ParamId};
    use crate::object::Fragment;
    use crate::semantics::actions::tests::collect_semantic_actions;
    use crate::session::builder::mock::{BackendEvent, MockSymbolId};
    use crate::session::builder::Width;
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
            BackendEvent::EmitFragment(Fragment::Immediate(
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
            BackendEvent::EmitFragment(Fragment::Immediate(
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
            BackendEvent::EmitFragment(Fragment::LdInlineAddr(
                0xf0,
                Expr::from_atom(Atom::Name(MockSymbolId(0)), "const".into()),
            ))
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
