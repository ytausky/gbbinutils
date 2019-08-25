use super::{BuiltinInstrSemantics, BuiltinInstrState, OperandSymbol, Session};

use crate::analyze::resolve::{NameTable, ResolvedName};
use crate::analyze::semantics::Keyword;
use crate::analyze::session::ReentrancyActions;
use crate::analyze::syntax::actions::*;
use crate::analyze::{IdentSource, Literal, StringSource};
use crate::diag::span::{Source, SpanSource};
use crate::diag::{Diagnostics, EmitDiag, Message};
use crate::expr::BinOp;

use std::ops::DerefMut;

pub(super) type ArgSemantics<R, N> = Session<
    R,
    N,
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

impl<R: ReentrancyActions, N> ArgFinalizer for ArgSemantics<R, N> {
    type Next = BuiltinInstrSemantics<R, N>;

    fn did_parse_arg(mut self) -> Self::Next {
        let arg = self.state.pop();
        self.state.parent.args.push(arg);
        assert_eq!(self.state.stack.len(), 0);
        self.map_line(|line| line.parent)
    }
}

impl<R, N> ArgActions<R::Ident, Literal<R::StringRef>, R::Span> for ArgSemantics<R, N>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: NameTable<
        R::Ident,
        Keyword = &'static Keyword,
        MacroId = R::MacroId,
        SymbolId = R::SymbolId,
    >,
{
    fn act_on_atom(&mut self, atom: ExprAtom<R::Ident, Literal<R::StringRef>>, span: R::Span) {
        self.state.stack.push(Arg {
            variant: ArgVariant::Atom(match atom {
                ExprAtom::Error => ArgAtom::Error,
                ExprAtom::Ident(ident) => match self.names.get(&ident) {
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

pub(super) type BuiltinInstrArgs<I, R, S> = Vec<Arg<I, R, S>>;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Arg<I, R, S> {
    pub variant: ArgVariant<I, R, S>,
    pub span: S,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ArgVariant<I, R, S> {
    Atom(ArgAtom<I, R>),
    Unary(ArgUnaryOp, Box<Arg<I, R, S>>),
    Binary(BinOp, Box<Arg<I, R, S>>, Box<Arg<I, R, S>>),
    FnCall((Option<I>, S), Vec<Arg<I, R, S>>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ArgAtom<I, R> {
    Error,
    Ident(I),
    Literal(Literal<R>),
    LocationCounter,
    OperandSymbol(OperandSymbol),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ArgUnaryOp {
    Parentheses,
}

impl<I, R> From<Literal<R>> for ArgAtom<I, R> {
    fn from(literal: Literal<R>) -> Self {
        ArgAtom::Literal(literal)
    }
}

#[cfg(test)]
impl<I, R, S> Arg<I, R, S> {
    pub fn from_atom<T: Into<ArgVariant<I, R, S>>>(atom: T, span: S) -> Self {
        Arg {
            variant: atom.into(),
            span,
        }
    }
}

impl<I, R, S> From<ArgAtom<I, R>> for ArgVariant<I, R, S> {
    fn from(atom: ArgAtom<I, R>) -> Self {
        ArgVariant::Atom(atom)
    }
}

impl<I, R, S: Clone> SpanSource for Arg<I, R, S> {
    type Span = S;
}

impl<I, R, S: Clone> Source for Arg<I, R, S> {
    fn span(&self) -> Self::Span {
        self.span.clone()
    }
}
