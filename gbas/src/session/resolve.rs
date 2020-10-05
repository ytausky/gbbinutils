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
    fn resolve_name_with_visibility(
        &mut self,
        ident: &I,
        visibility: Visibility,
    ) -> Option<ResolvedName<Self::MacroId, Self::SymbolId>>;

    fn define_name_with_visibility(
        &mut self,
        ident: I,
        visibility: Visibility,
        entry: ResolvedName<Self::MacroId, Self::SymbolId>,
    );
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedName<MacroId, SymbolId> {
    Keyword(&'static Keyword),
    Macro(MacroId),
    Symbol(SymbolId),
}

pub trait StartScope {
    fn start_scope(&mut self);
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
    fn resolve_name_with_visibility(
        &mut self,
        ident: &II::StringRef,
        visibility: Visibility,
    ) -> Option<ResolvedName<Self::MacroId, Self::SymbolId>> {
        let table = self.names.select_table_mut(visibility);
        let interner = &mut self.interner;
        table.get(&ident).cloned().map_or_else(
            || {
                let representative =
                    interner.intern(&interner.get_string(ident).to_ascii_uppercase());
                if let Some(keyword @ ResolvedName::Keyword(_)) =
                    table.get(&representative).cloned()
                {
                    table.insert(ident.clone(), keyword.clone());
                    Some(keyword)
                } else {
                    None
                }
            },
            Some,
        )
    }

    fn define_name_with_visibility(
        &mut self,
        ident: II::StringRef,
        visibility: Visibility,
        entry: ResolvedName<Self::MacroId, Self::SymbolId>,
    ) {
        let table = self.names.select_table_mut(visibility);
        table.insert(ident, entry);
    }
}

impl<C, R: SpanSource, II, M, B, D> StartScope
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
    fn start_scope(&mut self) {
        self.names.local = HashMap::new();
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::session::lex::StringSource;

    use log::Log;

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
        fn resolve_name_with_visibility(
            &mut self,
            ident: &String,
            visibility: Visibility,
        ) -> Option<ResolvedName<Self::MacroId, Self::SymbolId>> {
            let table = self.names.names.select_table_mut(visibility);
            table.get(ident).cloned().map_or_else(
                || {
                    let representative = ident.to_ascii_uppercase();
                    if let Some(keyword @ ResolvedName::Keyword(_)) =
                        table.get(&representative).cloned()
                    {
                        table.insert(ident.clone(), keyword.clone());
                        Some(keyword)
                    } else {
                        None
                    }
                },
                Some,
            )
        }

        fn define_name_with_visibility(
            &mut self,
            ident: String,
            visibility: Visibility,
            entry: ResolvedName<Self::MacroId, Self::SymbolId>,
        ) {
            let table = self.names.names.select_table_mut(visibility);
            table.insert(ident.clone(), entry.clone());
            self.names.log.push(NameTableEvent::Insert(ident, entry))
        }
    }

    impl<C, R, I, M, B, D, T> StartScope
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
        fn start_scope(&mut self) {
            self.names.names.local = HashMap::new();
            self.names.log.push(NameTableEvent::StartScope)
        }
    }

    #[derive(Debug, PartialEq)]
    pub enum NameTableEvent<MacroId, SymbolId> {
        Insert(String, ResolvedName<MacroId, SymbolId>),
        StartScope,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::semantics::actions::tests::TestOperation;
    use crate::session::builder::mock::MockSymbolId;
    use crate::session::mock::MockSession;

    use log::Log;

    #[test]
    fn retrieve_global_name() {
        let name = "start";
        let mut session = MockSession::<TestOperation<()>, ()>::new(Log::default());
        session.define_name_with_visibility(
            name.into(),
            Visibility::Global,
            ResolvedName::Symbol(MockSymbolId(42)),
        );
        assert_eq!(
            session.resolve_name_with_visibility(&name.into(), Visibility::Global),
            Some(ResolvedName::Symbol(MockSymbolId(42)))
        )
    }

    #[test]
    fn retrieve_local_name() {
        let mut session = MockSession::<TestOperation<()>, ()>::new(Log::default());
        session.start_scope();
        session.define_name_with_visibility(
            "_local".into(),
            Visibility::Local,
            ResolvedName::Symbol(MockSymbolId(42)),
        );
        assert_eq!(
            session.resolve_name_with_visibility(&"_local".into(), Visibility::Local),
            Some(ResolvedName::Symbol(MockSymbolId(42)))
        )
    }

    #[test]
    fn local_name_not_accessible_after_new_global_name() {
        let mut session = MockSession::<TestOperation<()>, ()>::new(Log::default());
        session.start_scope();
        session.define_name_with_visibility(
            "_local".into(),
            Visibility::Local,
            ResolvedName::Symbol(MockSymbolId(42)),
        );
        session.start_scope();
        assert_eq!(
            session.resolve_name_with_visibility(&"_local".into(), Visibility::Local),
            None
        )
    }
}
