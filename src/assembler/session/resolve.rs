use super::*;

use crate::span::SpanSource;

use std::collections::HashMap;

#[cfg(test)]
use crate::expr::{Atom, ExprOp};

pub(in crate::assembler) trait StartScope {
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

pub struct BiLevelNameTable<R> {
    pub(super) global: HashMap<R, NameEntry>,
    local: HashMap<R, NameEntry>,
}

impl<R> BiLevelNameTable<R> {
    pub fn new() -> Self {
        BiLevelNameTable {
            global: HashMap::new(),
            local: HashMap::new(),
        }
    }

    fn select_table_mut(&mut self, visibility: Visibility) -> &mut HashMap<R, NameEntry> {
        match visibility {
            Visibility::Global => &mut self.global,
            Visibility::Local => &mut self.local,
        }
    }
}

impl<C, R: SpanSystem, D> NameTable<StringRef> for CompositeSession<C, R, D>
where
    R: SpanSource + StripSpan<<R as SpanSource>::Span>,
{
    fn resolve_name_with_visibility(
        &mut self,
        ident: &StringRef,
        visibility: Visibility,
    ) -> Option<NameEntry> {
        let table = self.names.select_table_mut(visibility);
        table.get(ident).cloned().map_or_else(
            || {
                let representative = ident.as_ref().to_ascii_uppercase();
                if let Some(keyword @ NameEntry::OperandKeyword(_)) =
                    table.get(representative.as_str()).cloned()
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
        ident: StringRef,
        visibility: Visibility,
        entry: NameEntry,
    ) {
        #[cfg(test)]
        self.log_event(Event::DefineNameWithVisibility {
            ident: ident.clone(),
            visibility,
            entry: entry.clone(),
        });

        let table = self.names.select_table_mut(visibility);
        table.insert(ident, entry);
    }
}

impl<C, R, D> StartScope for CompositeSession<C, R, D>
where
    R: SpanSystem,
{
    fn start_scope(&mut self) {
        #[cfg(test)]
        self.log_event(Event::StartScope);

        self.names.local = HashMap::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::assembler::session::mock::MockSession;
    use crate::object::{Symbol, UserDefId};

    #[test]
    fn retrieve_global_name() {
        let name = "start";
        let entry = NameEntry::Symbol(Symbol::UserDef(UserDefId(42)));
        let mut session = MockSession::<()>::default();
        session.define_name_with_visibility(name.into(), Visibility::Global, entry.clone());
        assert_eq!(
            session.resolve_name_with_visibility(&name.into(), Visibility::Global),
            Some(entry)
        )
    }

    #[test]
    fn retrieve_local_name() {
        let entry = NameEntry::Symbol(Symbol::UserDef(UserDefId(42)));
        let mut session = MockSession::<()>::default();
        session.start_scope();
        session.define_name_with_visibility("_local".into(), Visibility::Local, entry.clone());
        assert_eq!(
            session.resolve_name_with_visibility(&"_local".into(), Visibility::Local),
            Some(entry)
        )
    }

    #[test]
    fn local_name_not_accessible_after_new_global_name() {
        let mut session = MockSession::<()>::default();
        session.start_scope();
        session.define_name_with_visibility(
            "_local".into(),
            Visibility::Local,
            NameEntry::Symbol(Symbol::UserDef(UserDefId(42))),
        );
        session.start_scope();
        assert_eq!(
            session.resolve_name_with_visibility(&"_local".into(), Visibility::Local),
            None
        )
    }
}
