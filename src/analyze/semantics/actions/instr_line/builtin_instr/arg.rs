use super::{BuiltinInstrSemantics, BuiltinInstrState, Session};

use crate::analyze::reentrancy::ReentrancyActions;
use crate::analyze::semantics::actions::Keyword;
use crate::analyze::semantics::arg::*;
use crate::analyze::semantics::resolve::{NameTable, ResolvedName};
use crate::analyze::syntax::actions::*;
use crate::analyze::{IdentSource, Literal, StringSource};
use crate::diag::span::SpanSource;
use crate::diag::{Diagnostics, EmitDiag, Message};
use crate::object::builder::SymbolSource;

use std::ops::DerefMut;

pub(in crate::analyze::semantics) type ArgSemantics<R, N, B> = Session<
    R,
    N,
    B,
    ExprBuilder<
        <R as IdentSource>::Ident,
        <R as StringSource>::StringRef,
        <R as SpanSource>::Span,
        BuiltinInstrState<R>,
    >,
>;

pub(crate) struct ExprBuilder<I, R, S, P> {
    stack: Vec<Arg<I, R, S>>,
    parent: P,
}

impl<I, R, S, P> ExprBuilder<I, R, S, P> {
    pub fn new(parent: P) -> Self {
        Self {
            stack: Vec::new(),
            parent,
        }
    }

    fn pop(&mut self) -> Arg<I, R, S> {
        self.stack.pop().unwrap_or_else(|| unreachable!())
    }
}

delegate_diagnostics! {
    {I, R, S, P: Diagnostics<S>}, ExprBuilder<I, R, S, P>, {parent}, P, S
}

impl<R: ReentrancyActions, N, B> ArgFinalizer for ArgSemantics<R, N, B> {
    type Next = BuiltinInstrSemantics<R, N, B>;

    fn did_parse_arg(mut self) -> Self::Next {
        let arg = self.state.pop();
        self.state.parent.args.push(arg);
        assert_eq!(self.state.stack.len(), 0);
        self.map_state(|line| line.parent)
    }
}

impl<R, N, B> ArgActions<R::Ident, Literal<R::StringRef>, R::Span> for ArgSemantics<R, N, B>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: NameTable<
        R::Ident,
        Keyword = &'static Keyword,
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
                    Some(ResolvedName::Keyword(_)) => unimplemented!(),
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
