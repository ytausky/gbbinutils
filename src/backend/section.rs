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

impl<SR> Node<SR> {
    fn len(&self) -> Value {
        match self {
            Node::Byte(_) => 1.into(),
            Node::Expr(_, width) => width.len().into(),
            Node::LdInlineAddr(..) => Value { min: 2, max: 3 },
        }
    }
}

fn evaluate_node_size<SR>(node: &Node<SR>, symbols: &SymbolTable) -> Value {
    match node {
        Node::Byte(_) => 1.into(),
        Node::Expr(_, width) => width.len().into(),
        Node::LdInlineAddr(expr, _) => match symbols.evaluate_expr_value(expr) {
            Some(Value { min, .. }) if min >= 0xff00 => 2.into(),
            Some(Value { max, .. }) if max < 0xff00 => 3.into(),
            _ => Value { min: 2, max: 3 },
        },
    }
}

impl<R> PendingSection<R> {
    pub fn new() -> PendingSection<R> {
        PendingSection {
            items: Vec::new(),
            location: 0.into(),
        }
    }

    pub fn push(&mut self, item: Node<R>) {
        self.location += item.len();
        self.items.push(item)
    }
}
