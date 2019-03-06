use std::collections::HashMap;

#[cfg(test)]
pub use self::mock::*;

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

pub trait StartScope<I> {
    fn start_scope(&mut self, ident: &I);
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

    fn select_table(&self, ident: &Ident<String>) -> &BasicNameTable<M, S> {
        match ident.visibility {
            Visibility::Global => &self.global,
            Visibility::Local => self.local.as_ref().unwrap(),
        }
    }

    fn select_table_mut(&mut self, ident: &Ident<String>) -> &mut BasicNameTable<M, S> {
        match ident.visibility {
            Visibility::Global => &mut self.global,
            Visibility::Local => self.local.as_mut().unwrap(),
        }
    }
}

impl<M, S> NameTable<Ident<String>> for BiLevelNameTable<M, S> {
    type MacroEntry = M;
    type SymbolEntry = S;

    fn get(&self, ident: &Ident<String>) -> Option<&Name<Self::MacroEntry, Self::SymbolEntry>> {
        self.select_table(ident).get(ident)
    }

    fn insert(&mut self, ident: Ident<String>, entry: Name<Self::MacroEntry, Self::SymbolEntry>) {
        self.select_table_mut(&ident).insert(ident, entry)
    }
}

impl<M, S> StartScope<Ident<String>> for BiLevelNameTable<M, S> {
    fn start_scope(&mut self, ident: &Ident<String>) {
        if ident.visibility == Visibility::Global {
            self.local.replace(BasicNameTable::new());
        }
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use std::cell::RefCell;

    pub struct MockNameTable<'a, N, T> {
        names: N,
        log: &'a RefCell<Vec<T>>,
    }

    impl<'a, N, T> MockNameTable<'a, N, T> {
        pub fn new(names: N, log: &'a RefCell<Vec<T>>) -> Self {
            Self { names, log }
        }
    }

    impl<'a, N: NameTable<Ident<String>>, T> NameTable<Ident<String>> for MockNameTable<'a, N, T> {
        type MacroEntry = N::MacroEntry;
        type SymbolEntry = N::SymbolEntry;

        fn get(&self, ident: &Ident<String>) -> Option<&Name<Self::MacroEntry, Self::SymbolEntry>> {
            self.names.get(ident)
        }

        fn insert(
            &mut self,
            ident: Ident<String>,
            entry: Name<Self::MacroEntry, Self::SymbolEntry>,
        ) {
            self.names.insert(ident, entry)
        }
    }

    impl<'a, N, T: From<NameTableEvent>> StartScope<Ident<String>> for MockNameTable<'a, N, T> {
        fn start_scope(&mut self, ident: &Ident<String>) {
            self.log
                .borrow_mut()
                .push(NameTableEvent::StartScope(ident.clone()).into())
        }
    }

    #[derive(Debug, PartialEq)]
    pub enum NameTableEvent {
        StartScope(Ident<String>),
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
        table.start_scope(&"global".into());
        table.insert("_local".into(), entry.clone());
        assert_eq!(table.get(&"_local".into()), Some(&entry))
    }

    #[test]
    fn local_name_not_accessible_after_new_global_name() {
        let mut table = BiLevelNameTable::<(), _>::new();
        table.start_scope(&"global1".into());
        table.insert("_local".into(), Name::Symbol(42));
        table.start_scope(&"global2".into());
        assert_eq!(table.get(&"_local".into()), None)
    }
}
