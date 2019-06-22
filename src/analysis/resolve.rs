use super::syntax::IdentFactory;

use std::collections::HashMap;

#[cfg(test)]
pub use self::mock::*;

#[cfg(test)]
use crate::model::{Atom, ExprOp};

pub trait NameTable<I> {
    type BackendEntry;
    type MacroEntry;

    fn get(&self, ident: &I) -> Option<&Name<Self::BackendEntry, Self::MacroEntry>>;
    fn insert(&mut self, ident: I, entry: Name<Self::BackendEntry, Self::MacroEntry>);
}

#[derive(Clone, Debug, PartialEq)]
pub enum Name<B, M> {
    Backend(B),
    Macro(M),
}

pub trait StartScope<I> {
    fn start_scope(&mut self, ident: &I);
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ident<T> {
    pub name: T,
    visibility: Visibility,
}

#[cfg(test)]
impl<L, T> From<Ident<T>> for Atom<L, Ident<T>> {
    fn from(ident: Ident<T>) -> Self {
        Atom::Name(ident)
    }
}

#[cfg(test)]
impl<L, T> From<Ident<T>> for ExprOp<Atom<L, Ident<T>>> {
    fn from(ident: Ident<T>) -> Self {
        Atom::from(ident).into()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Visibility {
    Global,
    Local,
}

pub struct DefaultIdentFactory;

impl IdentFactory for DefaultIdentFactory {
    type Ident = Ident<String>;

    fn mk_ident(&mut self, spelling: &str) -> Self::Ident {
        Self::Ident {
            name: spelling.to_string(),
            visibility: if spelling.starts_with('_') {
                Visibility::Local
            } else {
                Visibility::Global
            },
        }
    }
}

#[cfg(test)]
impl From<&str> for Ident<String> {
    fn from(name: &str) -> Ident<String> {
        DefaultIdentFactory.mk_ident(name)
    }
}

pub struct BasicNameTable<B, M> {
    table: HashMap<String, Name<B, M>>,
}

impl<B, M> BasicNameTable<B, M> {
    pub fn new() -> Self {
        BasicNameTable {
            table: HashMap::new(),
        }
    }
}

impl<B, M> NameTable<Ident<String>> for BasicNameTable<B, M> {
    type BackendEntry = B;
    type MacroEntry = M;

    fn get(&self, ident: &Ident<String>) -> Option<&Name<Self::BackendEntry, Self::MacroEntry>> {
        self.table.get(&ident.name)
    }

    fn insert(&mut self, ident: Ident<String>, entry: Name<Self::BackendEntry, Self::MacroEntry>) {
        self.table.insert(ident.name, entry);
    }
}

pub struct BiLevelNameTable<B, M> {
    global: BasicNameTable<B, M>,
    local: Option<BasicNameTable<B, M>>,
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

impl<B, M> NameTable<Ident<String>> for BiLevelNameTable<B, M> {
    type BackendEntry = B;
    type MacroEntry = M;

    fn get(&self, ident: &Ident<String>) -> Option<&Name<Self::BackendEntry, Self::MacroEntry>> {
        self.select_table(ident).get(ident)
    }

    fn insert(&mut self, ident: Ident<String>, entry: Name<Self::BackendEntry, Self::MacroEntry>) {
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

    use crate::log::Log;

    pub struct MockNameTable<N, T> {
        names: N,
        log: Log<T>,
    }

    impl<N, T> MockNameTable<N, T> {
        pub fn new(names: N, log: Log<T>) -> Self {
            Self { names, log }
        }
    }

    impl<N: NameTable<Ident<String>>, T> NameTable<Ident<String>> for MockNameTable<N, T> {
        type BackendEntry = N::BackendEntry;
        type MacroEntry = N::MacroEntry;

        fn get(
            &self,
            ident: &Ident<String>,
        ) -> Option<&Name<Self::BackendEntry, Self::MacroEntry>> {
            self.names.get(ident)
        }

        fn insert(
            &mut self,
            ident: Ident<String>,
            entry: Name<Self::BackendEntry, Self::MacroEntry>,
        ) {
            self.names.insert(ident, entry)
        }
    }

    impl<N, T: From<NameTableEvent>> StartScope<Ident<String>> for MockNameTable<N, T> {
        fn start_scope(&mut self, ident: &Ident<String>) {
            self.log.push(NameTableEvent::StartScope(ident.clone()))
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
        assert_eq!(
            DefaultIdentFactory.mk_ident("_loop").visibility,
            Visibility::Local
        )
    }

    #[test]
    fn ident_without_underscore_prefix_is_global() {
        assert_eq!(
            DefaultIdentFactory.mk_ident("start").visibility,
            Visibility::Global
        )
    }

    #[test]
    #[should_panic]
    fn panic_when_first_definition_is_local() {
        let mut table = BiLevelNameTable::<_, ()>::new();
        table.insert("_loop".into(), Name::Backend(()));
    }

    #[test]
    fn retrieve_global_name() {
        let name = "start";
        let mut table = BiLevelNameTable::<_, ()>::new();
        let entry = Name::Backend(42);
        table.insert(name.into(), entry.clone());
        assert_eq!(table.get(&name.into()), Some(&entry))
    }

    #[test]
    fn retrieve_local_name() {
        let mut table = BiLevelNameTable::<_, ()>::new();
        let entry = Name::Backend(42);
        table.start_scope(&"global".into());
        table.insert("_local".into(), entry.clone());
        assert_eq!(table.get(&"_local".into()), Some(&entry))
    }

    #[test]
    fn local_name_not_accessible_after_new_global_name() {
        let mut table = BiLevelNameTable::<_, ()>::new();
        table.start_scope(&"global1".into());
        table.insert("_local".into(), Name::Backend(42));
        table.start_scope(&"global2".into());
        assert_eq!(table.get(&"_local".into()), None)
    }
}
