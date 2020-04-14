use crate::analyze::Literal;
use crate::diag::span::{Source, SpanSource};
use crate::diag::{Diagnostics, Message};
use crate::expr::{BinOp, FnCall, LocationCounter};
use crate::object::builder::{Name, PushOp};

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) struct TreeArg<I, R, S> {
    pub variant: TreeArgVariant<I, R, S>,
    pub span: S,
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum TreeArgVariant<I, R, S> {
    Atom(TreeArgAtom<I, R>),
    Unary(ArgUnaryOp, Box<TreeArg<I, R, S>>),
    Binary(BinOp, Box<TreeArg<I, R, S>>, Box<TreeArg<I, R, S>>),
    FnCall((Option<I>, S), Vec<TreeArg<I, R, S>>),
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum TreeArgAtom<I, R> {
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

impl<I, R> From<Literal<R>> for TreeArgAtom<I, R> {
    fn from(literal: Literal<R>) -> Self {
        TreeArgAtom::Literal(literal)
    }
}

#[cfg(test)]
impl<I, R, S> TreeArg<I, R, S> {
    pub fn from_atom<T: Into<TreeArgVariant<I, R, S>>>(atom: T, span: S) -> Self {
        TreeArg {
            variant: atom.into(),
            span,
        }
    }
}

impl<I, R, S> From<TreeArgAtom<I, R>> for TreeArgVariant<I, R, S> {
    fn from(atom: TreeArgAtom<I, R>) -> Self {
        TreeArgVariant::Atom(atom)
    }
}

impl<I, R, S: Clone> SpanSource for TreeArg<I, R, S> {
    type Span = S;
}

impl<I, R, S: Clone> Source for TreeArg<I, R, S> {
    fn span(&self) -> Self::Span {
        self.span.clone()
    }
}

pub(super) trait EvalArg<I, R, S: Clone> {
    fn eval_arg(&mut self, arg: TreeArg<I, R, S>) -> Result<(), ()>;
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
    fn eval_arg(&mut self, arg: TreeArg<I, R, S>) -> Result<(), ()> {
        match arg.variant {
            TreeArgVariant::Atom(TreeArgAtom::Error) => return Err(()),
            TreeArgVariant::Atom(TreeArgAtom::Ident(ident)) => {
                self.push_op(Name(ident), arg.span);
                Ok(())
            }
            TreeArgVariant::Atom(TreeArgAtom::Literal(Literal::Number(n))) => {
                self.push_op(n, arg.span);
                Ok(())
            }
            TreeArgVariant::Atom(TreeArgAtom::OperandSymbol(_)) => Err(Message::KeywordInExpr {
                keyword: self.strip_span(&arg.span),
            }
            .at(arg.span)),
            TreeArgVariant::Atom(TreeArgAtom::Literal(Literal::String(_))) => {
                Err(Message::StringInInstruction.at(arg.span))
            }
            TreeArgVariant::Atom(TreeArgAtom::LocationCounter) => {
                self.push_op(LocationCounter, arg.span);
                Ok(())
            }
            TreeArgVariant::Unary(ArgUnaryOp::Parentheses, expr) => Ok(self.eval_arg(*expr)?),
            TreeArgVariant::Binary(binary, left, right) => {
                self.eval_arg(*left)?;
                self.eval_arg(*right)?;
                self.push_op(binary, arg.span);
                Ok(())
            }
            TreeArgVariant::FnCall((name, span), args) => {
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
