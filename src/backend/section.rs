use backend::{SymbolTable, Value};
use instruction::{Direction, RelocExpr};
use Width;

#[derive(Debug, PartialEq)]
pub enum Node<SR> {
    Byte(u8),
    Expr(RelocExpr<SR>, Width),
    LdInlineAddr(RelocExpr<SR>, Direction),
}

fn evaluate_node_size<'a, SR, D: 'a>(node: &Node<SR>, symbols: &'a SymbolTable<'a, D>) -> Value {
    match node {
        Node::Byte(_) => 1.into(),
        Node::Expr(_, width) => width.len().into(),
        Node::LdInlineAddr(expr, _) => {
            match symbols.evaluate_expr_value(expr) {
                Some(Value { min, .. }) if min >= 0xff00 => 2.into(),
                Some(Value { max, .. }) if max < 0xff00 => 3.into(),
                _ => Value { min: 2, max: 3 },
            }
        }
    }
}
