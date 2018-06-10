use backend::{SymbolTable, Value};
use instruction::{Direction, RelocExpr};
use Width;

pub struct PendingSection<R> {
    pub items: Vec<Node<R>>,
    pub location: Value,
}

#[derive(Debug, PartialEq)]
pub enum Node<SR> {
    Byte(u8),
    Expr(RelocExpr<SR>, Width),
    LdInlineAddr(RelocExpr<SR>, Direction),
}

impl<SR: Clone> Node<SR> {
    pub fn size(&self, symbols: &SymbolTable) -> Value {
        match self {
            Node::Byte(_) => 1.into(),
            Node::Expr(_, width) => width.len().into(),
            Node::LdInlineAddr(expr, _) => match expr.evaluate(symbols) {
                Ok(Value { min, .. }) if min >= 0xff00 => 2.into(),
                Ok(Value { max, .. }) if max < 0xff00 => 3.into(),
                _ => Value { min: 2, max: 3 },
            },
        }
    }
}

impl<R> PendingSection<R> {
    pub fn new() -> PendingSection<R> {
        PendingSection {
            items: Vec::new(),
            location: 0.into(),
        }
    }
}
