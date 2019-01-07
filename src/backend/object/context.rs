use super::resolve::Value;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

pub struct SymbolTable<I> {
    symbols: Vec<Value>,
    names: HashMap<I, SymbolId>,
    sizes: Vec<SymbolId>,
}

#[derive(Clone, Copy)]
pub struct SymbolId(usize);

pub trait Associate<I> {
    fn associate(self, context: &mut SymbolTable<I>, id: SymbolId);
}

pub trait ToSymbolId<I> {
    fn to_symbol_id(&self, table: &SymbolTable<I>) -> Option<SymbolId>;
}

impl<I> Associate<I> for SymbolId {
    fn associate(self, _: &mut SymbolTable<I>, _: SymbolId) {}
}

impl<I> ToSymbolId<I> for SymbolId {
    fn to_symbol_id(&self, _: &SymbolTable<I>) -> Option<SymbolId> {
        Some(*self)
    }
}

impl<I: Eq + Hash> Associate<I> for I {
    fn associate(self, context: &mut SymbolTable<I>, id: SymbolId) {
        context.names.insert(self, id);
    }
}

impl<I: Borrow<Q> + Eq + Hash, Q: Eq + Hash + ?Sized> ToSymbolId<I> for Q {
    fn to_symbol_id(&self, table: &SymbolTable<I>) -> Option<SymbolId> {
        table.names.get(self).cloned()
    }
}

#[derive(Clone, Copy)]
pub struct ChunkSize(pub usize);

impl<I> Associate<I> for ChunkSize {
    fn associate(self, context: &mut SymbolTable<I>, id: SymbolId) {
        let ChunkSize(index) = self;
        assert_eq!(index, context.sizes.len());
        context.sizes.push(id)
    }
}

impl<I> ToSymbolId<I> for ChunkSize {
    fn to_symbol_id(&self, context: &SymbolTable<I>) -> Option<SymbolId> {
        let ChunkSize(index) = *self;
        context.sizes.get(index).cloned()
    }
}

impl<I: Eq + Hash> SymbolTable<I> {
    pub fn new() -> SymbolTable<I> {
        SymbolTable {
            symbols: Vec::new(),
            names: HashMap::new(),
            sizes: Vec::new(),
        }
    }

    pub fn define(&mut self, key: impl Associate<I>, value: Value) {
        let id = SymbolId(self.symbols.len());
        self.symbols.push(value);
        key.associate(self, id)
    }

    pub fn get<K: ToSymbolId<I> + ?Sized>(&self, key: &K) -> Option<&Value> {
        key.to_symbol_id(self).map(|SymbolId(id)| &self.symbols[id])
    }

    fn get_mut(&mut self, key: &impl ToSymbolId<I>) -> Option<&mut Value> {
        key.to_symbol_id(self)
            .map(move |SymbolId(id)| &mut self.symbols[id])
    }

    pub fn refine(&mut self, key: &impl ToSymbolId<I>, value: Value) -> bool {
        let stored_value = self.get_mut(key).unwrap();
        let old_value = stored_value.clone();
        let was_refined = match (old_value, &value) {
            (Value::Unknown, new_value) => *new_value != Value::Unknown,
            (
                Value::Range {
                    min: old_min,
                    max: old_max,
                },
                Value::Range {
                    min: new_min,
                    max: new_max,
                },
            ) => {
                assert!(*new_min >= old_min);
                assert!(*new_max <= old_max);
                *new_min > old_min || *new_max < old_max
            }
            (Value::Range { .. }, Value::Unknown) => {
                panic!("a symbol previously approximated is now unknown")
            }
        };
        *stored_value = value;
        was_refined
    }
}

pub struct EvalContext<ST> {
    pub symbols: ST,
    pub location: Value,
}
