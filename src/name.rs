use std::collections::HashMap;

pub trait NameTable<I> {
    type MacroEntry;
    type SymbolEntry;

    fn get(&self, ident: &I) -> Option<&Name<Self::MacroEntry, Self::SymbolEntry>>;
    fn insert(&mut self, ident: I, entry: Name<Self::MacroEntry, Self::SymbolEntry>);
}

#[derive(Clone, Debug, PartialEq)]
pub enum Name<M, S> {
    Macro(M),
    Symbol(S),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ident<T> {
    pub name: T,
    visibility: Visibility,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Visibility {
    Global,
    Local,
}

pub fn mk_ident(spelling: &str) -> Ident<String> {
    Ident {
        name: spelling.to_string(),
        visibility: if spelling.starts_with('_') {
            Visibility::Local
        } else {
            Visibility::Global
        },
    }
}

#[cfg(test)]
impl From<&str> for Ident<String> {
    fn from(name: &str) -> Ident<String> {
        mk_ident(name)
    }
}

pub struct BasicNameTable<M, S> {
    table: HashMap<String, Name<M, S>>,
}

impl<M, S> BasicNameTable<M, S> {
    pub fn new() -> Self {
        BasicNameTable {
            table: HashMap::new(),
        }
    }
}

impl<M, S> NameTable<Ident<String>> for BasicNameTable<M, S> {
    type MacroEntry = M;
    type SymbolEntry = S;

    fn get(&self, ident: &Ident<String>) -> Option<&Name<Self::MacroEntry, Self::SymbolEntry>> {
        self.table.get(&ident.name)
    }

    fn insert(&mut self, ident: Ident<String>, entry: Name<Self::MacroEntry, Self::SymbolEntry>) {
        self.table.insert(ident.name, entry);
    }
}

pub struct BiLevelNameTable<M, S> {
    global: BasicNameTable<M, S>,
    local: Option<BasicNameTable<M, S>>,
}

impl<M, S> BiLevelNameTable<M, S> {
    pub fn new() -> Self {
        BiLevelNameTable {
            global: BasicNameTable::new(),
            local: None,
        }
    }
}

impl<M, S> NameTable<Ident<String>> for BiLevelNameTable<M, S> {
    type MacroEntry = M;
    type SymbolEntry = S;

    fn get(&self, ident: &Ident<String>) -> Option<&Name<Self::MacroEntry, Self::SymbolEntry>> {
        match ident.visibility {
            Visibility::Global => self.global.get(ident),
            Visibility::Local => self.local.as_ref().unwrap().get(ident),
        }
    }

    fn insert(&mut self, ident: Ident<String>, entry: Name<Self::MacroEntry, Self::SymbolEntry>) {
        match ident.visibility {
            Visibility::Global => {
                self.local.replace(BasicNameTable::new());
                self.global.insert(ident, entry)
            }
            Visibility::Local => self.local.as_mut().unwrap().insert(ident, entry),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ident_with_underscore_prefix_is_local() {
        assert_eq!(mk_ident("_loop").visibility, Visibility::Local)
    }

    #[test]
    fn ident_without_underscore_prefix_is_global() {
        assert_eq!(mk_ident("start").visibility, Visibility::Global)
    }

    #[test]
    #[should_panic]
    fn panic_when_first_definition_is_local() {
        let mut table = BiLevelNameTable::<(), _>::new();
        table.insert("_loop".into(), Name::Symbol(()));
    }

    #[test]
    fn retrieve_global_name() {
        let name = "start";
        let mut table = BiLevelNameTable::<(), _>::new();
        let entry = Name::Symbol(42);
        table.insert(name.into(), entry.clone());
        assert_eq!(table.get(&name.into()), Some(&entry))
    }

    #[test]
    fn retrieve_local_name() {
        let mut table = BiLevelNameTable::<(), _>::new();
        let entry = Name::Symbol(42);
        table.insert("global".into(), Name::Symbol(0));
        table.insert("_local".into(), entry.clone());
        assert_eq!(table.get(&"_local".into()), Some(&entry))
    }

    #[test]
    fn local_name_not_accessible_after_new_global_name() {
        let mut table = BiLevelNameTable::<(), _>::new();
        table.insert("global1".into(), Name::Symbol(0));
        table.insert("_local".into(), Name::Symbol(42));
        table.insert("global2".into(), Name::Symbol(1));
        assert_eq!(table.get(&"_local".into()), None)
    }
}
