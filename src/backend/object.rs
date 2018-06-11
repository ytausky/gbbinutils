use backend::{SymbolTable, Value};
use instruction::{Direction, RelocExpr};
use Width;

pub struct Object<SR> {
    pub sections: Vec<Section<SR>>,
}

pub struct Section<R> {
    pub items: Vec<Node<R>>,
}

#[derive(Debug, PartialEq)]
pub enum Node<SR> {
    Byte(u8),
    Expr(RelocExpr<SR>, Width),
    Label(String, SR),
    LdInlineAddr(RelocExpr<SR>, Direction),
}

impl<SR: Clone> Node<SR> {
    pub fn size(&self, symbols: &SymbolTable) -> Value {
        match self {
            Node::Byte(_) => 1.into(),
            Node::Expr(_, width) => width.len().into(),
            Node::Label(..) => 0.into(),
            Node::LdInlineAddr(expr, _) => match expr.evaluate(symbols) {
                Ok(Value { min, .. }) if min >= 0xff00 => 2.into(),
                Ok(Value { max, .. }) if max < 0xff00 => 3.into(),
                _ => Value { min: 2, max: 3 },
            },
        }
    }
}

impl<SR> Section<SR> {
    pub fn new() -> Section<SR> {
        Section { items: Vec::new() }
    }
}

pub struct ObjectBuilder<SR> {
    pub object: Object<SR>,
}

impl<SR> ObjectBuilder<SR> {
    pub fn new() -> ObjectBuilder<SR> {
        ObjectBuilder {
            object: Object {
                sections: vec![Section::new()],
            },
        }
    }
}
