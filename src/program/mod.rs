pub use self::builder::ProgramBuilder;

use self::context::{EvalContext, SymbolTable};
use self::resolve::Value;
use crate::backend::{BinaryObject, RelocExpr, Width};
use crate::diag::BackendDiagnostics;
use std::borrow::Borrow;

mod builder;
mod context;
mod lowering;
mod resolve;
mod translate;

#[derive(Clone, Copy)]
struct SymbolId(usize);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NameId(usize);

pub struct Program<S> {
    chunks: Vec<Chunk<S>>,
    symbols: SymbolTable,
}

struct Chunk<S> {
    origin: Option<RelocExpr<NameId, S>>,
    size: SymbolId,
    items: Vec<Node<S>>,
}

#[derive(Clone, Debug, PartialEq)]
enum Node<S> {
    Byte(u8),
    Expr(RelocExpr<NameId, S>, Width),
    LdInlineAddr(u8, RelocExpr<NameId, S>),
    Embedded(u8, RelocExpr<NameId, S>),
    Symbol((NameId, S), RelocExpr<NameId, S>),
}

impl<S> Program<S> {
    pub fn new() -> Program<S> {
        Program {
            chunks: Vec::new(),
            symbols: SymbolTable::new(),
        }
    }

    fn add_chunk(&mut self) {
        let size_symbol_id = self.symbols.new_symbol(Value::Unknown);
        self.chunks.push(Chunk::new(size_symbol_id))
    }
}

impl<S: Clone> Program<S> {
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
