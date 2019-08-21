use super::syntax::{IdentFactory, IdentSource};

use crate::object::builder::SymbolSource;

use std::collections::HashMap;

#[cfg(test)]
pub use self::mock::*;

#[cfg(test)]
use crate::expr::{Atom, ExprOp};

pub trait NameTable<I>: SymbolSource {
    type Keyword: Clone;
    type MacroId: Clone;

    fn get(&self, ident: &I) -> Option<ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>>;
    fn insert(
        &mut self,
        ident: I,
        entry: ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>,
    );
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedName<Keyword, MacroId, SymbolId> {
    Keyword(Keyword),
    Macro(MacroId),
    Symbol(SymbolId),
}

pub trait StartScope<I> {
    fn start_scope(&mut self, ident: &I);
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ident<T> {
    pub name: T,
    visibility: Visibility,
}

impl<T: AsRef<str>> AsRef<str> for Ident<T> {
    fn as_ref(&self) -> &str {
        self.name.as_ref()
    }
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

impl IdentSource for DefaultIdentFactory {
    type Ident = Ident<String>;
}

impl IdentFactory for DefaultIdentFactory {
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

pub struct BasicNameTable<Keyword, MacroId, SymbolId> {
    table: HashMap<String, ResolvedName<Keyword, MacroId, SymbolId>>,
}

impl<Keyword, MacroId, SymbolId> BasicNameTable<Keyword, MacroId, SymbolId> {
    pub fn new() -> Self {
        BasicNameTable {
            table: HashMap::new(),
        }
    }
}

impl<Keyword, MacroId, SymbolId> SymbolSource for BasicNameTable<Keyword, MacroId, SymbolId>
where
    Keyword: Clone,
    MacroId: Clone,
    SymbolId: Clone,
{
    type SymbolId = SymbolId;
}

impl<Keyword, MacroId, SymbolId> NameTable<String> for BasicNameTable<Keyword, MacroId, SymbolId>
where
    Keyword: Clone,
    MacroId: Clone,
    SymbolId: Clone,
{
    type Keyword = Keyword;
    type MacroId = MacroId;

    fn get(
        &self,
        ident: &String,
    ) -> Option<ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>> {
        self.table.get(ident).cloned()
    }

    fn insert(
        &mut self,
        ident: String,
        entry: ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>,
    ) {
        self.table.insert(ident, entry);
    }
}

pub struct BiLevelNameTable<Keyword, MacroId, SymbolId> {
    global: BasicNameTable<Keyword, MacroId, SymbolId>,
    local: Option<BasicNameTable<Keyword, MacroId, SymbolId>>,
}

impl<Keyword, MacroId, SymbolId> BiLevelNameTable<Keyword, MacroId, SymbolId> {
    pub fn new() -> Self {
        BiLevelNameTable {
            global: BasicNameTable::new(),
            local: None,
        }
    }

    fn select_table(&self, ident: &Ident<String>) -> &BasicNameTable<Keyword, MacroId, SymbolId> {
        match ident.visibility {
            Visibility::Global => &self.global,
            Visibility::Local => self.local.as_ref().unwrap(),
        }
    }

    fn select_table_mut(
        &mut self,
        ident: &Ident<String>,
    ) -> &mut BasicNameTable<Keyword, MacroId, SymbolId> {
        match ident.visibility {
            Visibility::Global => &mut self.global,
            Visibility::Local => self.local.as_mut().unwrap(),
        }
    }
}

impl<Keyword, MacroId, SymbolId> SymbolSource for BiLevelNameTable<Keyword, MacroId, SymbolId>
where
    SymbolId: Clone,
{
    type SymbolId = SymbolId;
}

impl<Keyword, MacroId, SymbolId> NameTable<Ident<String>>
    for BiLevelNameTable<Keyword, MacroId, SymbolId>
where
    Keyword: Clone,
    MacroId: Clone,
    SymbolId: Clone,
{
    type Keyword = Keyword;
    type MacroId = MacroId;

    fn get(
        &self,
        ident: &Ident<String>,
    ) -> Option<ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>> {
        self.select_table(ident).get(&ident.name)
    }

    fn insert(
        &mut self,
        ident: Ident<String>,
        entry: ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>,
    ) {
        self.select_table_mut(&ident).insert(ident.name, entry)
    }
}

impl<Keyword, MacroId, SymbolId> StartScope<Ident<String>>
    for BiLevelNameTable<Keyword, MacroId, SymbolId>
{
    fn start_scope(&mut self, ident: &Ident<String>) {
        if ident.visibility == Visibility::Global {
            self.local.replace(BasicNameTable::new());
        }
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::analyze::semantics::Keyword;
    use crate::log::Log;

    use std::marker::PhantomData;

    pub struct MockNameTable<N, T> {
        names: N,
        log: Log<T>,
    }

    impl<N, T> MockNameTable<N, T> {
        pub fn new(names: N, log: Log<T>) -> Self {
            Self { names, log }
        }
    }

    impl<N: SymbolSource, T> SymbolSource for MockNameTable<N, T> {
        type SymbolId = N::SymbolId;
    }

    impl<N, T> NameTable<String> for MockNameTable<N, T>
    where
        N: NameTable<String>,
        T: From<NameTableEvent<N::Keyword, N::MacroId, N::SymbolId>>,
    {
        type Keyword = N::Keyword;
        type MacroId = N::MacroId;

        fn get(
            &self,
            ident: &String,
        ) -> Option<ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>> {
            self.names.get(ident)
        }

        fn insert(
            &mut self,
            ident: String,
            entry: ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>,
        ) {
            self.names.insert(ident.clone(), entry.clone());
            self.log.push(NameTableEvent::Insert(ident, entry))
        }
    }

    impl<N, T> StartScope<String> for MockNameTable<N, T>
    where
        N: NameTable<String>,
        T: From<NameTableEvent<N::Keyword, N::MacroId, N::SymbolId>>,
    {
        fn start_scope(&mut self, ident: &String) {
            self.log.push(NameTableEvent::StartScope(ident.clone()))
        }
    }

    #[derive(Debug, PartialEq)]
    pub enum NameTableEvent<Keyword, MacroId, SymbolId> {
        Insert(String, ResolvedName<Keyword, MacroId, SymbolId>),
        StartScope(String),
    }

    pub(in crate::analyze) struct FakeNameTable<I>(PhantomData<I>);

    impl<I> FakeNameTable<I> {
        pub fn new() -> Self {
            Self(PhantomData)
        }
    }

    impl<I: Clone> SymbolSource for FakeNameTable<I> {
        type SymbolId = I;
    }

    impl<I: Clone> NameTable<I> for FakeNameTable<I> {
        type Keyword = Keyword;
        type MacroId = ();

        fn get(
            &self,
            ident: &I,
        ) -> Option<ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>> {
            Some(ResolvedName::Symbol(ident.clone()))
        }

        fn insert(&mut self, _: I, _: ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>) {
            panic!("tried to define a name")
        }
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
        let mut table = BiLevelNameTable::<(), (), _>::new();
        table.insert("_loop".into(), ResolvedName::Symbol(()));
    }

    #[test]
    fn retrieve_global_name() {
        let name = "start";
        let mut table = BiLevelNameTable::<(), (), _>::new();
        table.insert(name.into(), ResolvedName::Symbol(42));
        assert_eq!(table.get(&name.into()), Some(ResolvedName::Symbol(42)))
    }

    #[test]
    fn retrieve_local_name() {
        let mut table = BiLevelNameTable::<(), (), _>::new();
        table.start_scope(&"global".into());
        table.insert("_local".into(), ResolvedName::Symbol(42));
        assert_eq!(table.get(&"_local".into()), Some(ResolvedName::Symbol(42)))
    }

    #[test]
    fn local_name_not_accessible_after_new_global_name() {
        let mut table = BiLevelNameTable::<(), (), _>::new();
        table.start_scope(&"global1".into());
        table.insert("_local".into(), ResolvedName::Symbol(42));
        table.start_scope(&"global2".into());
        assert_eq!(table.get(&"_local".into()), None)
    }
}
