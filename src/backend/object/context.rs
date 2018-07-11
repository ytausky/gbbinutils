use super::resolve::Value;
use std::borrow::Borrow;
use std::collections::HashMap;

pub struct SymbolTable {
    symbols: Vec<Value>,
    names: HashMap<String, SymbolId>,
    sizes: Vec<SymbolId>,
}

#[derive(Clone, Copy)]
pub struct SymbolId(usize);

pub trait SymbolKey
where
    Self: Copy,
{
    fn associate(&self, context: &mut SymbolTable, id: SymbolId);
    fn to_symbol_id(&self, table: &SymbolTable) -> Option<SymbolId>;
}

impl SymbolKey for SymbolId {
    fn associate(&self, _: &mut SymbolTable, _: SymbolId) {}

    fn to_symbol_id(&self, _: &SymbolTable) -> Option<SymbolId> {
        Some(*self)
    }
}

impl<Q: Borrow<str> + Copy> SymbolKey for Q {
    fn associate(&self, context: &mut SymbolTable, id: SymbolId) {
        context.names.insert(self.borrow().to_string(), id);
    }

    fn to_symbol_id(&self, table: &SymbolTable) -> Option<SymbolId> {
        table.names.get(self.borrow()).cloned()
    }
}

#[derive(Clone, Copy)]
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

    pub fn define(&mut self, key: impl SymbolKey, value: Value) {
        let id = SymbolId(self.symbols.len());
        self.symbols.push(value);
        key.associate(self, id)
    }

    pub fn get(&self, key: impl SymbolKey) -> Option<&Value> {
        key.to_symbol_id(self).map(|SymbolId(id)| &self.symbols[id])
    }

    fn get_mut(&mut self, key: impl SymbolKey) -> Option<&mut Value> {
        key.to_symbol_id(self)
            .map(move |SymbolId(id)| &mut self.symbols[id])
    }

    pub fn refine(&mut self, key: impl SymbolKey, value: Value) -> bool {
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
