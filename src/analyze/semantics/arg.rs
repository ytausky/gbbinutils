use crate::analyze::Literal;
use crate::diag::span::{Source, SpanSource};
use crate::diag::{Diagnostics, Message};
use crate::expr::{BinOp, FnCall, LocationCounter};
use crate::object::builder::{Name, PushOp};

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) struct Arg<I, R, S> {
    pub variant: ArgVariant<I, R, S>,
    pub span: S,
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum ArgVariant<I, R, S> {
    Atom(ArgAtom<I, R>),
    Unary(ArgUnaryOp, Box<Arg<I, R, S>>),
    Binary(BinOp, Box<Arg<I, R, S>>, Box<Arg<I, R, S>>),
    FnCall((Option<I>, S), Vec<Arg<I, R, S>>),
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum ArgAtom<I, R> {
    Error,
    Ident(I),
    Literal(Literal<R>),
    LocationCounter,
    OperandSymbol(OperandSymbol),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OperandSymbol {
    A,
    Af,
    B,
    Bc,
    C,
    D,
    De,
    E,
    H,
    Hl,
    Hld,
    Hli,
    L,
    Nc,
    Nz,
    Sp,
    Z,
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum ArgUnaryOp {
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

pub(super) trait EvalArg<I, R, S: Clone> {
    fn eval_arg(&mut self, arg: Arg<I, R, S>) -> Result<(), ()>;
}

impl<'a, T, I, R, S> EvalArg<I, R, S> for T
where
    T: PushOp<LocationCounter, S>
        + PushOp<i32, S>
        + PushOp<Name<I>, S>
        + PushOp<BinOp, S>
        + PushOp<FnCall, S>
        + Diagnostics<S>,
    R: Eq,
    S: Clone,
{
    fn eval_arg(&mut self, arg: Arg<I, R, S>) -> Result<(), ()> {
        match arg.variant {
            ArgVariant::Atom(ArgAtom::Error) => Err(())?,
            ArgVariant::Atom(ArgAtom::Ident(ident)) => {
                self.push_op(Name(ident), arg.span);
                Ok(())
            }
            ArgVariant::Atom(ArgAtom::Literal(Literal::Number(n))) => {
                self.push_op(n, arg.span);
                Ok(())
            }
            ArgVariant::Atom(ArgAtom::OperandSymbol(_)) => Err(Message::KeywordInExpr {
                keyword: self.strip_span(&arg.span),
            }
            .at(arg.span)),
            ArgVariant::Atom(ArgAtom::Literal(Literal::String(_))) => {
                Err(Message::StringInInstruction.at(arg.span))
            }
            ArgVariant::Atom(ArgAtom::LocationCounter) => {
                self.push_op(LocationCounter, arg.span);
                Ok(())
            }
            ArgVariant::Unary(ArgUnaryOp::Parentheses, expr) => Ok(self.eval_arg(*expr)?),
            ArgVariant::Binary(binary, left, right) => {
                self.eval_arg(*left)?;
                self.eval_arg(*right)?;
                self.push_op(binary, arg.span);
                Ok(())
            }
            ArgVariant::FnCall((name, span), args) => {
                let n = args.len();
                for arg in args {
                    self.eval_arg(arg)?;
                }
                self.push_op(Name(name.ok_or(())?), span.clone());
                self.push_op(FnCall(n), span);
                Ok(())
            }
        }
        .map_err(|diagnostic| {
            self.emit_diag(diagnostic);
        })
    }
}
