use std::collections::HashMap;

pub trait NameTable<I> {
    type MacroEntry;
    type SymbolEntry;

    fn get(&self, ident: &I) -> Option<&Name<Self::MacroEntry, Self::SymbolEntry>>;
    fn insert(&mut self, ident: I, entry: Name<Self::MacroEntry, Self::SymbolEntry>);
}

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
impl<T> From<T> for Ident<T> {
    fn from(name: T) -> Ident<T> {
        Ident {
            name,
            visibility: Visibility::Global,
        }
    }
}

#[cfg(test)]
impl From<&str> for Ident<String> {
    fn from(name: &str) -> Ident<String> {
        Ident {
            name: name.into(),
            visibility: Visibility::Global,
        }
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
}
