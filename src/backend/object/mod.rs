use self::context::{EvalContext, SymbolTable};
use self::resolve::Value;
use backend::{BinaryObject, RelocExpr};
use diagnostics::{DiagnosticsListener, Span};
use std::borrow::Borrow;
use Width;

mod context;
mod resolve;
mod translate;

pub struct Object<SR> {
    chunks: Vec<Chunk<SR>>,
}

pub struct Chunk<R> {
    origin: Option<RelocExpr<R>>,
    items: Vec<Node<R>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Node<SR> {
    Byte(u8),
    Expr(RelocExpr<SR>, Width),
    Label(String, SR),
    LdInlineAddr(u8, RelocExpr<SR>),
    Embedded(u8, RelocExpr<SR>),
}

impl<SR> Object<SR> {
    pub fn new() -> Object<SR> {
        Object { chunks: Vec::new() }
    }

    fn add_chunk(&mut self) {
        self.chunks.push(Chunk::new())
    }
}

pub fn link<'a, S, D>(object: Object<S>, diagnostics: &D) -> BinaryObject
where
    S: Span,
    D: DiagnosticsListener<S> + 'a,
{
    let symbols = resolve::resolve_symbols(&object);
    let mut context = EvalContext {
        symbols: &symbols,
        location: 0.into(),
    };
    BinaryObject {
        sections: object
            .chunks
            .into_iter()
            .map(|chunk| chunk.translate(&mut context, diagnostics))
            .collect(),
    }
}

impl<SR> Chunk<SR> {
    pub fn new() -> Chunk<SR> {
        Chunk {
            origin: None,
            items: Vec::new(),
        }
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
                self.object.add_chunk();
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

    pub fn constrain_origin(&mut self, origin: RelocExpr<SR>) {
        self.state = Some(BuilderState::Pending {
            origin: Some(origin),
        })
    }
}

impl<S: Span> Chunk<S> {
    fn traverse<ST, F>(&self, context: &mut EvalContext<ST>, f: F) -> Value
    where
        ST: Borrow<SymbolTable>,
        F: FnMut(&Node<S>, &mut EvalContext<ST>),
    {
        context.location = self.evaluate_origin(context);
        traverse_chunk_items(&self.items, context, f)
    }

    fn evaluate_origin<ST: Borrow<SymbolTable>>(&self, context: &EvalContext<ST>) -> Value {
        self.origin
            .as_ref()
            .map(|expr| expr.evaluate(context))
            .unwrap_or_else(|| 0.into())
    }
}

fn traverse_chunk_items<S, ST, F>(
    items: &[Node<S>],
    context: &mut EvalContext<ST>,
    mut f: F,
) -> Value
where
    S: Span,
    ST: Borrow<SymbolTable>,
    F: FnMut(&Node<S>, &mut EvalContext<ST>),
{
    let origin = context.location.clone();
    let mut offset = Value::from(0);
    for item in items {
        offset += item.size(&context);
        context.location = origin.clone() + offset.clone();
        f(item, context)
    }
    offset
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::BinaryOperator;
    use diagnostics::IgnoreDiagnostics;

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
    fn constrain_origin_determines_origin_of_new_chunk() {
        let origin = RelocExpr::Literal(0x3000, ());
        let object = build_object(|builder| {
            builder.constrain_origin(origin.clone());
            builder.push(Node::Byte(0xcd))
        });
        assert_eq!(object.chunks[0].origin, Some(origin))
    }

    #[test]
    fn resolve_origin_relative_to_previous_chunk() {
        let origin1 = 0x150;
        let skipped_bytes = 0x10;
        let object = Object {
            chunks: vec![
                Chunk {
                    origin: Some(RelocExpr::Literal(origin1, ())),
                    items: vec![Node::Byte(0x42)],
                },
                Chunk {
                    origin: Some(RelocExpr::BinaryOperation(
                        Box::new(RelocExpr::LocationCounter(())),
                        Box::new(RelocExpr::Literal(skipped_bytes, ())),
                        BinaryOperator::Plus,
                        (),
                    )),
                    items: vec![Node::Byte(0x43)],
                },
            ],
        };
        let binary = link(object, &IgnoreDiagnostics {});
        assert_eq!(
            binary.sections[1].origin,
            (origin1 + 1 + skipped_bytes) as usize
        )
    }

    fn build_object(f: impl FnOnce(&mut ObjectBuilder<()>)) -> Object<()> {
        let mut builder = ObjectBuilder::new();
        f(&mut builder);
        builder.build()
    }
}
