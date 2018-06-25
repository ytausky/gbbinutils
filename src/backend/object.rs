use backend::{Data, LinkingContext, Value};
use diagnostics::{DiagnosticsListener, SourceInterval};
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
    Embedded(u8, RelocExpr<SR>),
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
    pub fn size(&self, symbols: &LinkingContext) -> Value {
        match self {
            Node::Byte(_) | Node::Embedded(..) => 1.into(),
            Node::Expr(_, width) => width.len().into(),
            Node::Label(..) => 0.into(),
            Node::LdInlineAddr(expr, _) => match expr.evaluate(symbols) {
                Ok(Some(Value { min, .. })) if min >= 0xff00 => 2.into(),
                Ok(Some(Value { max, .. })) if max < 0xff00 => 3.into(),
                _ => Value { min: 2, max: 3 },
            },
        }
    }
}

impl<SR: SourceInterval> Node<SR> {
    pub fn translate(
        &self,
        symbols: &LinkingContext,
        diagnostics: &impl DiagnosticsListener<SR>,
    ) -> impl Iterator<Item = Data> {
        match self {
            Node::Byte(value) => Some(Data::Byte(*value)),
            Node::Embedded(..) | Node::LdInlineAddr(..) => panic!(),
            Node::Expr(expr, width) => Some(
                symbols
                    .resolve_expr_item(&expr, *width)
                    .unwrap_or_else(|diagnostic| {
                        diagnostics.emit_diagnostic(diagnostic);
                        match width {
                            Width::Byte => Data::Byte(0),
                            Width::Word => Data::Word(0),
                        }
                    }),
            ),
            Node::Label(..) => None,
        }.into_iter()
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

    #[cfg(test)]
    fn set_origin(&mut self, origin: RelocExpr<SR>) {
        self.state = Some(BuilderState::Pending {
            origin: Some(origin),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_object_has_no_chunks() {
        let object = build_object(|_| ());
        assert_eq!(object.chunks.len(), 0)
    }

    #[test]
    fn no_origin_by_default() {
        let object = build_object(|builder| builder.push(Node::Byte(0xcd)));
        assert_eq!(object.chunks[0].origin, None)
    }

    #[test]
    fn set_origin_determines_origin_of_new_chunk() {
        let origin = RelocExpr::Literal(0x3000, ());
        let object = build_object(|builder| {
            builder.set_origin(origin.clone());
            builder.push(Node::Byte(0xcd))
        });
        assert_eq!(object.chunks[0].origin, Some(origin))
    }

    fn build_object(f: impl FnOnce(&mut ObjectBuilder<()>)) -> Object<()> {
        let mut builder = ObjectBuilder::new();
        f(&mut builder);
        builder.build()
    }
}
