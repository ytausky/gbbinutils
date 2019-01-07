use self::context::{EvalContext, SymbolTable};
use self::resolve::Value;
use crate::backend::{BinaryObject, RelocExpr, Width};
use crate::diag::BackendDiagnostics;
use std::borrow::Borrow;
use std::hash::Hash;

mod context;
mod resolve;
mod translate;

#[derive(Clone, Copy)]
pub struct SymbolId(usize);

pub struct Object<I, S> {
    chunks: Vec<Chunk<I, S>>,
    symbols: SymbolTable<I>,
}

pub(crate) struct Chunk<I, S> {
    origin: Option<RelocExpr<I, S>>,
    size: SymbolId,
    items: Vec<Node<I, S>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Node<I, S> {
    Byte(u8),
    Expr(RelocExpr<I, S>, Width),
    LdInlineAddr(u8, RelocExpr<I, S>),
    Embedded(u8, RelocExpr<I, S>),
    Symbol((I, S), RelocExpr<I, S>),
}

impl<I: Eq + Hash, S> Object<I, S> {
    pub fn new() -> Object<I, S> {
        Object {
            chunks: Vec::new(),
            symbols: SymbolTable::new(),
        }
    }

    fn add_chunk(&mut self) {
        let size_symbol_id = SymbolId(self.symbols.symbols.len());
        self.symbols.symbols.push(Value::Unknown);
        self.chunks.push(Chunk::new(size_symbol_id))
    }
}

impl<I: Clone + Eq + Hash, S: Clone> Object<I, S> {
    pub(crate) fn link(mut self, diagnostics: &mut impl BackendDiagnostics<S>) -> BinaryObject {
        self.resolve_symbols();
        let mut context = EvalContext {
            symbols: &self.symbols,
            location: 0.into(),
        };
        BinaryObject {
            sections: self
                .chunks
                .into_iter()
                .map(|chunk| chunk.translate(&mut context, diagnostics))
                .collect(),
        }
    }
}

impl<I, S> Chunk<I, S> {
    pub fn new(size: SymbolId) -> Chunk<I, S> {
        Chunk {
            origin: None,
            size,
            items: Vec::new(),
        }
    }
}

pub struct ObjectBuilder<SR> {
    object: Object<String, SR>,
    state: Option<BuilderState<SR>>,
}

enum BuilderState<SR> {
    Pending {
        origin: Option<RelocExpr<String, SR>>,
    },
    InChunk(usize),
}

impl<SR> ObjectBuilder<SR> {
    pub fn new() -> ObjectBuilder<SR> {
        ObjectBuilder {
            object: Object::new(),
            state: Some(BuilderState::Pending { origin: None }),
        }
    }

    pub fn push(&mut self, node: Node<String, SR>) {
        self.current_chunk().items.push(node)
    }

    pub fn build(self) -> Object<String, SR> {
        self.object
    }

    fn current_chunk(&mut self) -> &mut Chunk<String, SR> {
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

    pub fn constrain_origin(&mut self, origin: RelocExpr<String, SR>) {
        self.state = Some(BuilderState::Pending {
            origin: Some(origin),
        })
    }
}

impl<I: Eq + Hash, S: Clone> Chunk<I, S> {
    fn traverse<ST, F>(&self, context: &mut EvalContext<ST>, f: F) -> Value
    where
        ST: Borrow<SymbolTable<I>>,
        F: FnMut(&Node<I, S>, &mut EvalContext<ST>),
    {
        context.location = self.evaluate_origin(context);
        traverse_chunk_items(&self.items, context, f)
    }

    fn evaluate_origin<ST: Borrow<SymbolTable<I>>>(&self, context: &EvalContext<ST>) -> Value {
        self.origin
            .as_ref()
            .map(|expr| expr.evaluate(context))
            .unwrap_or_else(|| 0.into())
    }
}

fn traverse_chunk_items<I, S, ST, F>(
    items: &[Node<I, S>],
    context: &mut EvalContext<ST>,
    mut f: F,
) -> Value
where
    I: Eq + Hash,
    S: Clone,
    ST: Borrow<SymbolTable<I>>,
    F: FnMut(&Node<I, S>, &mut EvalContext<ST>),
{
    let origin = context.location.clone();
    let mut offset = Value::from(0);
    for item in items {
        offset += &item.size(&context);
        context.location = &origin + &offset;
        f(item, context)
    }
    offset
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BinaryOperator, RelocAtom};
    use crate::diag::IgnoreDiagnostics;
    use crate::expr::ExprVariant;

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
        let origin: RelocExpr<_, _> = 0x3000.into();
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
        let object = Object::<String, _> {
            chunks: vec![
                Chunk {
                    origin: Some(origin1.into()),
                    size: SymbolId(0),
                    items: vec![Node::Byte(0x42)],
                },
                Chunk {
                    origin: Some(
                        ExprVariant::Binary(
                            BinaryOperator::Plus,
                            Box::new(RelocAtom::LocationCounter.into()),
                            Box::new(skipped_bytes.into()),
                        )
                        .into(),
                    ),
                    size: SymbolId(1),
                    items: vec![Node::Byte(0x43)],
                },
            ],
            symbols: {
                let mut table = SymbolTable::new();
                table.symbols.push(Value::Unknown);
                table.symbols.push(Value::Unknown);
                table
            },
        };
        let binary = object.link(&mut IgnoreDiagnostics::new());
        assert_eq!(
            binary.sections[1].origin,
            (origin1 + 1 + skipped_bytes) as usize
        )
    }

    fn build_object(f: impl FnOnce(&mut ObjectBuilder<()>)) -> Object<String, ()> {
        let mut builder = ObjectBuilder::new();
        f(&mut builder);
        builder.build()
    }
}
