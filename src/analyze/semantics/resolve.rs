use crate::analyze::macros::MacroSource;
use crate::analyze::syntax::{IdentFactory, IdentSource};
use crate::object::builder::SymbolSource;

use std::collections::HashMap;

#[cfg(test)]
pub use self::mock::*;

#[cfg(test)]
use crate::expr::{Atom, ExprOp};

pub(in crate::analyze) trait NameTable<I>: MacroSource + SymbolSource {
    type Keyword: Clone;

    fn resolve_name(
        &mut self,
        ident: &I,
    ) -> Option<ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>>;

    fn define_name(
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

impl From<&str> for Ident<String> {
    fn from(name: &str) -> Ident<String> {
        DefaultIdentFactory.mk_ident(name)
    }
}

pub struct BasicNameTable<Keyword, MacroId, SymbolId> {
    table: HashMap<String, ResolvedName<Keyword, MacroId, SymbolId>>,
}

impl<Keyword, MacroId, SymbolId> Default for BasicNameTable<Keyword, MacroId, SymbolId> {
    fn default() -> Self {
        Self {
            table: HashMap::default(),
        }
    }
}

impl<Keyword, MacroId, SymbolId> MacroSource for BasicNameTable<Keyword, MacroId, SymbolId>
where
    Keyword: Clone,
    MacroId: Clone,
    SymbolId: Clone,
{
    type MacroId = MacroId;
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

    fn resolve_name(
        &mut self,
        ident: &String,
    ) -> Option<ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>> {
        self.table.get(ident).cloned().map_or_else(
            || {
                let repr = ident.to_ascii_uppercase();
                if let Some(keyword @ ResolvedName::Keyword(_)) = self.table.get(&repr).cloned() {
                    self.table.insert(repr, keyword.clone());
                    Some(keyword)
                } else {
                    None
                }
            },
            Some,
        )
    }

    fn define_name(
        &mut self,
        ident: String,
        entry: ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>,
    ) {
        self.table.insert(ident, entry);
    }
}

pub struct BiLevelNameTable<T> {
    global: T,
    local: Option<T>,
}

impl<T: Default> BiLevelNameTable<T> {
    pub fn new() -> Self {
        BiLevelNameTable {
            global: Default::default(),
            local: None,
        }
    }

    fn select_table_mut(&mut self, ident: &Ident<String>) -> &mut T {
        match ident.visibility {
            Visibility::Global => &mut self.global,
            Visibility::Local => self.local.as_mut().unwrap(),
        }
    }
}

impl<T: MacroSource> MacroSource for BiLevelNameTable<T> {
    type MacroId = T::MacroId;
}

impl<T: SymbolSource> SymbolSource for BiLevelNameTable<T> {
    type SymbolId = T::SymbolId;
}

impl<T: Default + NameTable<String>> NameTable<Ident<String>> for BiLevelNameTable<T> {
    type Keyword = T::Keyword;

    fn resolve_name(
        &mut self,
        ident: &Ident<String>,
    ) -> Option<ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>> {
        self.select_table_mut(ident).resolve_name(&ident.name)
    }

    fn define_name(
        &mut self,
        ident: Ident<String>,
        entry: ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>,
    ) {
        self.select_table_mut(&ident).define_name(ident.name, entry)
    }
}

impl<T: Default> StartScope<Ident<String>> for BiLevelNameTable<T> {
    fn start_scope(&mut self, ident: &Ident<String>) {
        if ident.visibility == Visibility::Global {
            self.local.replace(Default::default());
        }
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::log::Log;

    pub(in crate::analyze) struct MockNameTable<N, T> {
        names: N,
        log: Log<T>,
    }

    impl<N, T> MockNameTable<N, T> {
        pub fn new(names: N, log: Log<T>) -> Self {
            Self { names, log }
        }
    }

    impl<N: MacroSource, T> MacroSource for MockNameTable<N, T> {
        type MacroId = N::MacroId;
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

        fn resolve_name(
            &mut self,
            ident: &String,
        ) -> Option<ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>> {
            self.names.resolve_name(ident)
        }

        fn define_name(
            &mut self,
            ident: String,
            entry: ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>,
        ) {
            self.names.define_name(ident.clone(), entry.clone());
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
        let mut table = BiLevelNameTable::<BasicNameTable<(), (), _>>::new();
        table.define_name("_loop".into(), ResolvedName::Symbol(()));
    }

    #[test]
    fn retrieve_global_name() {
        let name = "start";
        let mut table = BiLevelNameTable::<BasicNameTable<(), (), _>>::new();
        table.define_name(name.into(), ResolvedName::Symbol(42));
        assert_eq!(
            table.resolve_name(&name.into()),
            Some(ResolvedName::Symbol(42))
        )
    }

    #[test]
    fn retrieve_local_name() {
        let mut table = BiLevelNameTable::<BasicNameTable<(), (), _>>::new();
        table.start_scope(&"global".into());
        table.define_name("_local".into(), ResolvedName::Symbol(42));
        assert_eq!(
            table.resolve_name(&"_local".into()),
            Some(ResolvedName::Symbol(42))
        )
    }

    #[test]
    fn local_name_not_accessible_after_new_global_name() {
        let mut table = BiLevelNameTable::<BasicNameTable<(), (), _>>::new();
        table.start_scope(&"global1".into());
        table.define_name("_local".into(), ResolvedName::Symbol(42));
        table.start_scope(&"global2".into());
        assert_eq!(table.resolve_name(&"_local".into()), None)
    }
}