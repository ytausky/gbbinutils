use super::resolve::Value;
use super::{NameId, SymbolId};

pub struct SymbolTable {
    pub symbols: Vec<Value>,
    pub names: Vec<Option<SymbolId>>,
}

pub trait ToSymbolId: Copy {
    fn to_symbol_id(self, table: &SymbolTable) -> Option<SymbolId>;
}

impl ToSymbolId for SymbolId {
    fn to_symbol_id(self, _: &SymbolTable) -> Option<SymbolId> {
        Some(self)
    }
}

impl ToSymbolId for NameId {
    fn to_symbol_id(self, table: &SymbolTable) -> Option<SymbolId> {
        let NameId(name_id) = self;
        table.names[name_id]
    }
}

impl SymbolTable {
    pub fn new() -> SymbolTable {
        SymbolTable {
            symbols: Vec::new(),
            names: Vec::new(),
        }
    }

    pub fn get<K: ToSymbolId>(&self, key: K) -> Option<&Value> {
        key.to_symbol_id(self).map(|SymbolId(id)| &self.symbols[id])
    }

    fn get_mut(&mut self, key: impl ToSymbolId) -> Option<&mut Value> {
        key.to_symbol_id(self)
            .map(move |SymbolId(id)| &mut self.symbols[id])
    }

    pub fn refine(&mut self, key: impl ToSymbolId, value: Value) -> bool {
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
