use super::macros::MacroSource;
use super::{CompositeSession, Interner};

use crate::semantics::Keyword;
use crate::session::builder::SymbolSource;
use crate::span::SpanSource;

use std::collections::HashMap;

#[cfg(test)]
pub use self::mock::*;

#[cfg(test)]
use crate::expr::{Atom, ExprOp};

pub(crate) trait NameTable<I>: MacroSource + SymbolSource {
    fn resolve_name(&mut self, ident: &I) -> Option<ResolvedName<Self::MacroId, Self::SymbolId>>;
    fn define_name(&mut self, ident: I, entry: ResolvedName<Self::MacroId, Self::SymbolId>);
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedName<MacroId, SymbolId> {
    Keyword(&'static Keyword),
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
impl<T> From<Ident<T>> for Atom<Ident<T>> {
    fn from(ident: Ident<T>) -> Self {
        Atom::Name(ident)
    }
}

#[cfg(test)]
impl<T> From<Ident<T>> for ExprOp<Ident<T>> {
    fn from(ident: Ident<T>) -> Self {
        Atom::from(ident).into()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Visibility {
    Global,
    Local,
}

pub struct DefaultIdentFactory;

impl DefaultIdentFactory {
    pub fn mk_ident(&mut self, spelling: &str) -> Ident<String> {
        Ident {
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

pub struct BiLevelNameTable<M, S, R> {
    pub global: HashMap<R, ResolvedName<M, S>>,
    local: HashMap<R, ResolvedName<M, S>>,
}

impl<M, S, R> BiLevelNameTable<M, S, R> {
    pub fn new() -> Self {
        BiLevelNameTable {
            global: HashMap::new(),
            local: HashMap::new(),
        }
    }

    fn select_table_mut(&mut self, visibility: Visibility) -> &mut HashMap<R, ResolvedName<M, S>> {
        match visibility {
            Visibility::Global => &mut self.global,
            Visibility::Local => &mut self.local,
        }
    }
}

impl<M: Clone, S, R> MacroSource for BiLevelNameTable<M, S, R> {
    type MacroId = M;
}

impl<M, S: Clone, R> SymbolSource for BiLevelNameTable<M, S, R> {
    type SymbolId = S;
}

impl<C, R: SpanSource, II, M, B, D> NameTable<II::StringRef>
    for CompositeSession<
        C,
        R,
        II,
        M,
        BiLevelNameTable<M::MacroId, B::SymbolId, II::StringRef>,
        B,
        D,
    >
where
    II: Interner,
    M: MacroSource,
    B: SymbolSource,
{
    fn resolve_name(
        &mut self,
        ident: &II::StringRef,
    ) -> Option<ResolvedName<Self::MacroId, Self::SymbolId>> {
        let string = DefaultIdentFactory.mk_ident(self.interner.get_string(ident));
        let table = self.names.select_table_mut(string.visibility);
        let interner = &mut self.interner;
        table.get(&ident).cloned().map_or_else(
            || {
                let representative = interner.intern(&string.name.to_ascii_uppercase());
                if let Some(keyword @ ResolvedName::Keyword(_)) =
                    table.get(&representative).cloned()
                {
                    table.insert(representative, keyword.clone());
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
        ident: II::StringRef,
        entry: ResolvedName<Self::MacroId, Self::SymbolId>,
    ) {
        let string = DefaultIdentFactory.mk_ident(self.interner.get_string(&ident));
        let table = self.names.select_table_mut(string.visibility);
        table.insert(ident, entry);
    }
}

impl<C, R: SpanSource, II, M, B, D> StartScope<II::StringRef>
    for CompositeSession<
        C,
        R,
        II,
        M,
        BiLevelNameTable<M::MacroId, B::SymbolId, II::StringRef>,
        B,
        D,
    >
where
    II: Interner,
    M: MacroSource,
    B: SymbolSource,
{
    fn start_scope(&mut self, ident: &II::StringRef) {
        let ident = DefaultIdentFactory.mk_ident(self.interner.get_string(&ident));
        if ident.visibility == Visibility::Global {
            self.names.local = HashMap::new();
        }
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::log::Log;
    use crate::session::lex::StringSource;

    pub(crate) struct MockNameTable<N, T> {
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

    impl<C, R, I, M, B, D, T> NameTable<String>
        for CompositeSession<
            C,
            R,
            I,
            M,
            MockNameTable<BiLevelNameTable<M::MacroId, B::SymbolId, I::StringRef>, T>,
            B,
            D,
        >
    where
        R: SpanSource,
        I: StringSource<StringRef = String>,
        M: MacroSource,
        B: SymbolSource,
        T: From<NameTableEvent<M::MacroId, B::SymbolId>>,
    {
        fn resolve_name(
            &mut self,
            ident: &String,
        ) -> Option<ResolvedName<Self::MacroId, Self::SymbolId>> {
            let ident = DefaultIdentFactory.mk_ident(ident);
            let table = self.names.names.select_table_mut(ident.visibility);
            table.get(&ident.name).cloned().map_or_else(
                || {
                    let repr = ident.name.to_ascii_uppercase();
                    if let Some(keyword @ ResolvedName::Keyword(_)) = table.get(&repr).cloned() {
                        table.insert(ident.name.clone(), keyword.clone());
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
            entry: ResolvedName<Self::MacroId, Self::SymbolId>,
        ) {
            let table = self
                .names
                .names
                .select_table_mut(DefaultIdentFactory.mk_ident(&ident).visibility);
            table.insert(ident.clone(), entry.clone());
            self.names.log.push(NameTableEvent::Insert(ident, entry))
        }
    }

    impl<C, R, I, M, B, D, T> StartScope<String>
        for CompositeSession<
            C,
            R,
            I,
            M,
            MockNameTable<BiLevelNameTable<M::MacroId, B::SymbolId, I::StringRef>, T>,
            B,
            D,
        >
    where
        R: SpanSource,
        I: StringSource,
        M: MacroSource,
        B: SymbolSource,
        T: From<NameTableEvent<M::MacroId, B::SymbolId>>,
    {
        fn start_scope(&mut self, ident: &String) {
            let ident = DefaultIdentFactory.mk_ident(&ident);
            if ident.visibility == Visibility::Global {
                self.names.names.local = HashMap::new();
            }
            self.names.log.push(NameTableEvent::StartScope(ident.name))
        }
    }

    #[derive(Debug, PartialEq)]
    pub enum NameTableEvent<MacroId, SymbolId> {
        Insert(String, ResolvedName<MacroId, SymbolId>),
        StartScope(String),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::log::Log;
    use crate::semantics::actions::tests::TestOperation;
    use crate::session::builder::mock::MockSymbolId;
    use crate::session::mock::MockSession;

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
    fn retrieve_global_name() {
        let name = "start";
        let mut session = MockSession::<TestOperation<()>, ()>::new(Log::new());
        session.define_name(name.into(), ResolvedName::Symbol(MockSymbolId(42)));
        assert_eq!(
            session.resolve_name(&name.into()),
            Some(ResolvedName::Symbol(MockSymbolId(42)))
        )
    }

    #[test]
    fn retrieve_local_name() {
        let mut session = MockSession::<TestOperation<()>, ()>::new(Log::new());
        session.start_scope(&"global".into());
        session.define_name("_local".into(), ResolvedName::Symbol(MockSymbolId(42)));
        assert_eq!(
            session.resolve_name(&"_local".into()),
            Some(ResolvedName::Symbol(MockSymbolId(42)))
        )
    }

    #[test]
    fn local_name_not_accessible_after_new_global_name() {
        let mut session = MockSession::<TestOperation<()>, ()>::new(Log::new());
        session.start_scope(&"global1".into());
        session.define_name("_local".into(), ResolvedName::Symbol(MockSymbolId(42)));
        session.start_scope(&"global2".into());
        assert_eq!(session.resolve_name(&"_local".into()), None)
    }
}
