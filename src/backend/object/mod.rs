use self::context::{EvalContext, SymbolTable};
use self::resolve::Value;
use crate::backend::{Backend, BinaryObject, Item, RelocExpr, Width};
use crate::diag::BackendDiagnostics;
use std::borrow::Borrow;
use std::collections::HashMap;

mod context;
mod lowering;
mod resolve;
mod translate;

#[derive(Clone, Copy)]
pub struct SymbolId(usize);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NameId(usize);

pub struct Object<S> {
    chunks: Vec<Chunk<S>>,
    symbols: SymbolTable,
}

pub(crate) struct Chunk<S> {
    origin: Option<RelocExpr<NameId, S>>,
    size: SymbolId,
    items: Vec<Node<S>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Node<S> {
    Byte(u8),
    Expr(RelocExpr<NameId, S>, Width),
    LdInlineAddr(u8, RelocExpr<NameId, S>),
    Embedded(u8, RelocExpr<NameId, S>),
    Symbol((NameId, S), RelocExpr<NameId, S>),
}

impl<S> Object<S> {
    pub fn new() -> Object<S> {
        Object {
            chunks: Vec::new(),
            symbols: SymbolTable::new(),
        }
    }

    fn add_chunk(&mut self) {
        let size_symbol_id = self.symbols.new_symbol(Value::Unknown);
        self.chunks.push(Chunk::new(size_symbol_id))
    }
}

impl<S: Clone> Object<S> {
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

impl<S> Chunk<S> {
    pub fn new(size: SymbolId) -> Chunk<S> {
        Chunk {
            origin: None,
            size,
            items: Vec::new(),
        }
    }
}

pub struct ObjectBuilder<SR> {
    object: Object<SR>,
    state: Option<BuilderState<SR>>,
    names: HashMap<String, NameId>,
}

enum BuilderState<SR> {
    Pending {
        origin: Option<RelocExpr<NameId, SR>>,
    },
    InChunk(usize),
}

impl<SR> ObjectBuilder<SR> {
    pub fn new() -> ObjectBuilder<SR> {
        ObjectBuilder {
            object: Object::new(),
            state: Some(BuilderState::Pending { origin: None }),
            names: HashMap::new(),
        }
    }

    pub fn push(&mut self, node: Node<SR>) {
        self.current_chunk().items.push(node)
    }

    pub fn build(self) -> Object<SR> {
        self.object
    }

    pub fn define(&mut self, (name, span): (String, SR), value: RelocExpr<NameId, SR>) {
        let name_id = self.lookup(name);
        self.object.symbols.define_name(name_id, Value::Unknown);
        self.push(Node::Symbol((name_id, span), value))
    }

    pub fn lookup(&mut self, name: String) -> NameId {
        let symbols = &mut self.object.symbols;
        *self.names.entry(name).or_insert_with(|| symbols.new_name())
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

    pub fn constrain_origin(&mut self, origin: RelocExpr<NameId, SR>) {
        self.state = Some(BuilderState::Pending {
            origin: Some(origin),
        })
    }
}

impl<S: Clone> Chunk<S> {
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
    S: Clone,
    ST: Borrow<SymbolTable>,
    F: FnMut(&Node<S>, &mut EvalContext<ST>),
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

impl<S: Clone + 'static> Backend<String, S> for ObjectBuilder<S> {
    type Object = Object<S>;

    fn define_symbol(&mut self, symbol: (String, S), value: Self::Value) {
        self.define(symbol, value)
    }

    fn emit_item(&mut self, item: Item<Self::Value>) {
        use self::lowering::Lower;
        item.lower().for_each(|data_item| self.push(data_item))
    }

    fn into_object(self) -> Self::Object {
        self.build()
    }

    fn set_origin(&mut self, origin: Self::Value) {
        self.constrain_origin(origin)
    }
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
        let object = Object {
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
                table.new_symbol(Value::Unknown);
                table.new_symbol(Value::Unknown);
                table
            },
        };
        let binary = object.link(&mut IgnoreDiagnostics::new());
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
