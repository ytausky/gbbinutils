use super::resolve::Value;
use super::{NameDef, NameId, ValueId};

pub(super) struct NameTable(Vec<Option<NameDef>>);

impl NameTable {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub(super) fn alloc_name(&mut self) -> NameId {
        let id = NameId(self.0.len());
        self.0.push(None);
        id
    }

    pub(super) fn define_name(&mut self, NameId(id): NameId, def: NameDef) {
        assert!(self.0[id].is_none());
        self.0[id] = Some(def);
    }

    pub(super) fn get_name_def(&self, NameId(id): NameId) -> Option<&NameDef> {
        self.0[id].as_ref()
    }
}

pub(super) struct RelocTable {
    values: Vec<Value>,
}

impl RelocTable {
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    pub fn alloc(&mut self, value: Value) -> ValueId {
        let id = ValueId(self.values.len());
        self.values.push(value);
        id
    }

    pub(super) fn get_value(&self, ValueId(id): ValueId) -> Value {
        self.values[id].clone()
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
}

pub(super) struct EvalContext<'a, R> {
    pub names: &'a NameTable,
    pub relocs: R,
    pub location: Value,
}
