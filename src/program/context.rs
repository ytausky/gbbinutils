use super::resolve::Value;
use super::{NameDef, NameId, ValueId};

pub(super) struct SymbolTable {
    names: Vec<Option<NameDef>>,
    values: Vec<Value>,
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

    pub(super) fn get_value(&self, ValueId(id): ValueId) -> Value {
        self.values[id].clone()
    }

    pub(super) fn alloc_name(&mut self) -> NameId {
        let id = NameId(self.names.len());
        self.names.push(None);
        id
    }

    pub(super) fn define_name(&mut self, NameId(id): NameId, def: NameDef) {
        assert!(self.names[id].is_none());
        self.names[id] = Some(def);
    }

    pub(super) fn get_name_def(&self, NameId(id): NameId) -> Option<&NameDef> {
        self.names[id].as_ref()
    }

    pub fn refine(&mut self, ValueId(id): ValueId, value: Value) -> bool {
        let stored_value = &mut self.values[id];
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
