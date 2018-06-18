use backend::{SymbolTable, Value};
use instruction::{Direction, RelocExpr};
use Width;

pub struct Object<SR> {
    pub chunks: Vec<Chunk<SR>>,
}

pub struct Chunk<R> {
    name: String,
    origin: Option<RelocExpr<R>>,
    pub items: Vec<Node<R>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Node<SR> {
    Byte(u8),
    Expr(RelocExpr<SR>, Width),
    Label(String, SR),
    LdInlineAddr(RelocExpr<SR>, Direction),
}

impl<SR> Object<SR> {
    pub fn new() -> Object<SR> {
        Object { chunks: Vec::new() }
    }

    pub fn add_chunk(&mut self, name: impl Into<String>) {
        self.chunks.push(Chunk::new(name.into()))
    }
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

impl<SR> Chunk<SR> {
    pub fn new(name: impl Into<String>) -> Chunk<SR> {
        Chunk {
            name: name.into(),
            origin: None,
            items: Vec::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

pub struct ObjectBuilder<SR> {
    object: Object<SR>,
    state: Option<BuilderState<SR>>,
}

enum BuilderState<SR> {
    Pending { origin: Option<RelocExpr<SR>> },
    InChunk(usize),
}

impl<SR> ObjectBuilder<SR> {
    pub fn new() -> ObjectBuilder<SR> {
        ObjectBuilder {
            object: Object::new(),
            state: Some(BuilderState::Pending { origin: None }),
        }
    }

    pub fn push(&mut self, node: Node<SR>) {
        self.current_chunk().items.push(node)
    }

    pub fn build(self) -> Object<SR> {
        self.object
    }

    fn current_chunk(&mut self) -> &mut Chunk<SR> {
        match self.state.take().unwrap() {
            BuilderState::Pending { origin } => {
                self.object.add_chunk("name");
                let index = self.object.chunks.len() - 1;
                self.state = Some(BuilderState::InChunk(index));
                let chunk = &mut self.object.chunks[index];
                chunk.origin = origin;
                chunk
            }
            BuilderState::InChunk(index) => {
                self.state = Some(BuilderState::InChunk(index));
                &mut self.object.chunks[index]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_object_has_no_chunks() {
        let object = ObjectBuilder::<()>::new().build();
        assert_eq!(object.chunks.len(), 0)
    }

    #[test]
    fn no_origin_by_default() {
        let mut builder = ObjectBuilder::<()>::new();
        builder.push(Node::Byte(0xcd));
        let object = builder.build();
        assert_eq!(object.chunks[0].origin, None)
    }
}
