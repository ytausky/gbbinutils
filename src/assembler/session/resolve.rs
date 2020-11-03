use super::*;

use std::collections::HashMap;

pub struct BiLevelNameTable<R> {
    pub(super) global: HashMap<R, NameEntry>,
    pub(super) local: HashMap<R, NameEntry>,
}

impl<R> BiLevelNameTable<R> {
    pub fn new() -> Self {
        BiLevelNameTable {
            global: HashMap::new(),
            local: HashMap::new(),
        }
    }

    fn select_table_mut(&mut self, ident: &str) -> &mut HashMap<R, NameEntry> {
        if ident.starts_with('_') {
            &mut self.local
        } else {
            &mut self.global
        }
    }
}

impl<'a, R: SpanSystem> IdentTable for CompositeSession<'a, R> {
    fn query_term(&mut self, ident: &StringRef) -> NameEntry {
        let table = self.names.select_table_mut(&ident);
        if let Some(entry) = table.get(ident) {
            entry.clone()
        } else {
            let representative = ident.to_ascii_uppercase();
            if let Some(entry @ NameEntry::OperandKeyword(_)) = table.get(representative.as_str()) {
                let entry = entry.clone();
                table.insert(ident.clone(), entry.clone());
                entry.clone()
            } else {
                let symbol = self.builder.alloc_symbol(ident.clone());
                let entry = NameEntry::Symbol(symbol);
                table.insert(ident.clone(), entry.clone());
                entry
            }
        }
    }
}
