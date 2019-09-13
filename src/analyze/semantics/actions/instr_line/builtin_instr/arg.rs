use super::BuiltinInstrSemantics;

use crate::analyze::reentrancy::ReentrancyActions;
use crate::analyze::semantics::actions::Keyword;
use crate::analyze::semantics::arg::*;
use crate::analyze::semantics::builtin_instr::BuiltinInstrSet;
use crate::analyze::semantics::resolve::{NameTable, ResolvedName};
use crate::analyze::semantics::{ArgSemantics, ExprBuilder};
use crate::analyze::syntax::actions::*;
use crate::analyze::Literal;
use crate::diag::{Diagnostics, EmitDiag, Message};
use crate::object::builder::SymbolSource;

use std::ops::DerefMut;

delegate_diagnostics! {
    {I, R, S, P: Diagnostics<S>}, ExprBuilder<I, R, S, P>, {parent}, P, S
}

impl<I, R, N, B> ArgFinalizer for ArgSemantics<I, R, N, B>
where
    I: BuiltinInstrSet<R>,
    R: ReentrancyActions,
{
    type Next = BuiltinInstrSemantics<I, R, N, B>;

    fn did_parse_arg(mut self) -> Self::Next {
        let arg = self.state.pop();
        self.state.parent.args.push(arg);
        assert_eq!(self.state.stack.len(), 0);
        self.map_state(|line| line.parent)
    }
}

impl<I, R, N, B> ArgActions<R::Ident, Literal<R::StringRef>, R::Span> for ArgSemantics<I, R, N, B>
where
    I: BuiltinInstrSet<R>,
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: NameTable<
        R::Ident,
        Keyword = &'static Keyword<I::Binding, I::Free>,
        MacroId = R::MacroId,
        SymbolId = B::SymbolId,
    >,
    B: SymbolSource,
{
    fn act_on_atom(&mut self, atom: ExprAtom<R::Ident, Literal<R::StringRef>>, span: R::Span) {
        self.state.stack.push(Arg {
            variant: ArgVariant::Atom(match atom {
                ExprAtom::Error => ArgAtom::Error,
                ExprAtom::Ident(ident) => match self.names.resolve_name(&ident) {
                    Some(ResolvedName::Keyword(Keyword::Operand(operand))) => {
                        ArgAtom::OperandSymbol(*operand)
                    }
                    Some(ResolvedName::Keyword(_)) => {
                        let keyword = self.reentrancy.strip_span(&span);
                        self.reentrancy
                            .emit_diag(Message::KeywordInExpr { keyword }.at(span.clone()));
                        ArgAtom::Error
                    }
                    _ => ArgAtom::Ident(ident),
                },
                ExprAtom::Literal(literal) => ArgAtom::Literal(literal),
                ExprAtom::LocationCounter => ArgAtom::LocationCounter,
            }),
            span,
        })
    }

    fn act_on_operator(&mut self, op: Operator, span: R::Span) {
        let variant = match op {
            Operator::Unary(UnaryOperator::Parentheses) => {
                let inner = self.state.pop();
                ArgVariant::Unary(ArgUnaryOp::Parentheses, Box::new(inner))
            }
            Operator::Binary(binary) => {
                let rhs = self.state.pop();
                let lhs = self.state.pop();
                ArgVariant::Binary(binary, Box::new(lhs), Box::new(rhs))
            }
            Operator::FnCall(n) => {
                let args = self.state.stack.split_off(self.state.stack.len() - n);
                let name = self.state.pop();
                let name = (
                    match name.variant {
                        ArgVariant::Atom(ArgAtom::Ident(ident)) => Some(ident),
                        _ => {
                            self.emit_diag(Message::OnlyIdentsCanBeCalled.at(name.span.clone()));
                            None
                        }
                    },
                    name.span,
                );
                ArgVariant::FnCall(name, args)
            }
        };
        self.state.stack.push(Arg { variant, span })
    }
}

#[cfg(test)]
mod tests {
    use crate::analyze::semantics::actions::tests::collect_semantic_actions;
    use crate::analyze::syntax::actions::*;
    use crate::diag::{DiagnosticsEvent, Message, MockSpan};

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
}
