use super::resolve::Value;
use super::{NameId, ValueId};

pub(super) struct SymbolTable {
    names: Vec<Option<NameDef>>,
    values: Vec<Value>,
}

enum NameDef {
    Value(ValueId),
}

pub(super) trait ToValueId: Copy {
    fn to_value_id(self, table: &SymbolTable) -> Option<ValueId>;
}

impl ToValueId for ValueId {
    fn to_value_id(self, _: &SymbolTable) -> Option<ValueId> {
        Some(self)
    }
}

impl ToValueId for NameId {
    fn to_value_id(self, table: &SymbolTable) -> Option<ValueId> {
        let NameId(name_id) = self;
        table.names[name_id].as_ref().map(|body| match body {
            NameDef::Value(id) => *id,
        })
    }
}

impl SymbolTable {
    pub fn new() -> SymbolTable {
        SymbolTable {
            names: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn new_symbol(&mut self, value: Value) -> ValueId {
        let id = ValueId(self.values.len());
        self.values.push(value);
        id
    }

    pub(super) fn alloc_name(&mut self) -> NameId {
        let id = NameId(self.names.len());
        self.names.push(None);
        id
    }

    pub(super) fn define_name(&mut self, NameId(id): NameId, value: Value) {
        assert!(self.names[id].is_none());
        let symbol_id = self.new_symbol(value);
        self.names[id] = Some(NameDef::Value(symbol_id));
    }

    pub fn get<K: ToValueId>(&self, key: K) -> Option<&Value> {
        key.to_value_id(self).map(|ValueId(id)| &self.values[id])
    }

    fn get_mut(&mut self, key: impl ToValueId) -> Option<&mut Value> {
        key.to_value_id(self)
            .map(move |ValueId(id)| &mut self.values[id])
    }

    pub fn refine(&mut self, key: impl ToValueId, value: Value) -> bool {
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

    #[cfg(test)]
    pub fn names(&self) -> impl Iterator<Item = Option<&Value>> {
        self.names.iter().map(move |entry| {
            entry
                .as_ref()
                .map(|NameDef::Value(ValueId(id))| &self.values[*id])
        })
    }
}

pub struct EvalContext<ST> {
    pub symbols: ST,
    pub location: Value,
}
