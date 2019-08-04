use super::{BuiltinInstrSemantics, OperandSymbol};

use crate::analysis::session::Session;
use crate::analysis::syntax::{ArgActions, ArgFinalizer, ExprAtom, Operator, UnaryOperator};
use crate::analysis::Literal;
use crate::diag::span::{Source, SpanSource};
use crate::diag::{Diagnostics, EmitDiag, Message};
use crate::model::BinOp;

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

impl<S: Session> ArgFinalizer
    for ExprBuilder<S::Ident, S::StringRef, S::Span, BuiltinInstrSemantics<S>>
{
    type Next = BuiltinInstrSemantics<S>;

    fn did_parse_arg(mut self) -> Self::Next {
        if !self.parent.has_errors {
            assert_eq!(self.stack.len(), 1);
            self.parent.args.push(self.stack.pop().unwrap());
        }
        self.parent
    }
}

impl<I, R, S, P> ArgActions<S> for ExprBuilder<I, R, S, P>
where
    I: AsRef<str>,
    S: Clone,
    Self: Diagnostics<S>,
{
    type Ident = I;
    type Literal = Literal<R>;

    fn act_on_atom(&mut self, atom: (ExprAtom<Self::Ident, Self::Literal>, S)) {
        self.stack.push(Arg {
            variant: ArgVariant::Atom(match atom.0 {
                ExprAtom::Error => ArgAtom::Error,
                ExprAtom::Ident(ident) => OPERAND_SYMBOLS
                    .iter()
                    .find(|(spelling, _)| spelling.eq_ignore_ascii_case(ident.as_ref()))
                    .map(|(_, symbol)| ArgAtom::OperandSymbol(*symbol))
                    .unwrap_or_else(|| ArgAtom::Ident(ident)),
                ExprAtom::Literal(literal) => ArgAtom::Literal(literal),
                ExprAtom::LocationCounter => ArgAtom::LocationCounter,
            }),
            span: atom.1,
        })
    }

    fn act_on_operator(&mut self, (op, span): (Operator, S)) {
        let variant = match op {
            Operator::Unary(UnaryOperator::Parentheses) => {
                let inner = self.pop();
                ArgVariant::Unary(ArgUnaryOp::Parentheses, Box::new(inner))
            }
            Operator::Binary(binary) => {
                let rhs = self.pop();
                let lhs = self.pop();
                ArgVariant::Binary(binary, Box::new(lhs), Box::new(rhs))
            }
            Operator::FnCall(n) => {
                let args = self.stack.split_off(self.stack.len() - n);
                let name = self.pop();
                let name = (
                    match name.variant {
                        ArgVariant::Atom(ArgAtom::Ident(ident)) => ident,
                        _ => {
                            self.emit_diag(Message::OnlyIdentsCanBeCalled.at(name.span));
                            return;
                        }
                    },
                    name.span,
                );
                ArgVariant::FnCall(name, args)
            }
        };
        self.stack.push(Arg { variant, span })
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
    FnCall((I, S), Vec<Arg<I, R, S>>),
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

const OPERAND_SYMBOLS: &[(&str, OperandSymbol)] = &[
    ("a", OperandSymbol::A),
    ("af", OperandSymbol::Af),
    ("b", OperandSymbol::B),
    ("bc", OperandSymbol::Bc),
    ("c", OperandSymbol::C),
    ("d", OperandSymbol::D),
    ("de", OperandSymbol::De),
    ("e", OperandSymbol::E),
    ("h", OperandSymbol::H),
    ("hl", OperandSymbol::Hl),
    ("hld", OperandSymbol::Hld),
    ("hli", OperandSymbol::Hli),
    ("l", OperandSymbol::L),
    ("nc", OperandSymbol::Nc),
    ("nz", OperandSymbol::Nz),
    ("sp", OperandSymbol::Sp),
    ("z", OperandSymbol::Z),
];
