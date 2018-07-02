use backend::link::Value;
use std::borrow::Borrow;
use std::collections::HashMap;

pub struct SymbolTable {
    symbols: Vec<Option<Value>>,
    names: HashMap<String, SymbolId>,
    sizes: Vec<SymbolId>,
}

#[derive(Clone, Copy)]
pub struct SymbolId(usize);

pub trait SymbolKey {
    fn associate(&self, context: &mut SymbolTable, id: SymbolId);
    fn to_symbol_id(&self, table: &SymbolTable) -> Option<SymbolId>;
}

impl SymbolKey for SymbolId {
    fn associate(&self, _: &mut SymbolTable, _: SymbolId) {}

    fn to_symbol_id(&self, _: &SymbolTable) -> Option<SymbolId> {
        Some((*self).clone())
    }
}

impl<Q: Borrow<str>> SymbolKey for Q {
    fn associate(&self, context: &mut SymbolTable, id: SymbolId) {
        context.names.insert(self.borrow().to_string(), id);
    }

    fn to_symbol_id(&self, table: &SymbolTable) -> Option<SymbolId> {
        table.names.get(self.borrow()).cloned()
    }
}

pub struct ChunkSize(pub usize);

impl SymbolKey for ChunkSize {
    fn associate(&self, context: &mut SymbolTable, id: SymbolId) {
        let ChunkSize(index) = *self;
        assert_eq!(index, context.sizes.len());
        context.sizes.push(id)
    }

    fn to_symbol_id(&self, context: &SymbolTable) -> Option<SymbolId> {
        let ChunkSize(index) = *self;
        context.sizes.get(index).cloned()
    }
}

impl SymbolTable {
    pub fn new() -> SymbolTable {
        SymbolTable {
            symbols: Vec::new(),
            names: HashMap::new(),
            sizes: Vec::new(),
        }
    }

    pub fn define(&mut self, key: impl SymbolKey, value: Option<Value>) {
        let id = SymbolId(self.symbols.len());
        self.symbols.push(value);
        key.associate(self, id)
    }

    pub fn get(&self, key: impl SymbolKey) -> Option<&Option<Value>> {
        key.to_symbol_id(self).map(|SymbolId(id)| &self.symbols[id])
    }

    fn get_mut(&mut self, key: impl SymbolKey) -> Option<&mut Option<Value>> {
        key.to_symbol_id(self)
            .map(move |SymbolId(id)| &mut self.symbols[id])
    }

    pub fn refine(&mut self, key: impl SymbolKey, value: Value) -> bool {
        let stored_value = self.get_mut(key).unwrap();
        let old_value = stored_value.clone();
        let was_refined = old_value.map_or(true, |v| value.len() < v.len());
        *stored_value = Some(value);
        was_refined
    }
}

pub struct EvalContext<'a> {
    pub symbols: &'a SymbolTable,
    pub location: Option<Value>,
}
