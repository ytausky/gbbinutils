use self::num::Num;

use crate::expr::{Atom, ExprOp};
use crate::span::SpanSource;

use std::ops::{Index, IndexMut};

pub mod num;

pub struct Object<M: SpanSource> {
    pub data: Data<M::Span>,
    pub metadata: M,
}

pub struct Data<S> {
    pub content: Content<S>,
    pub vars: VarTable,
}

pub struct Content<S> {
    pub sections: Vec<Section<S>>,
    pub symbols: SymbolTable<S>,
}

pub struct Section<S> {
    pub constraints: Constraints<S>,
    pub addr: VarId,
    pub size: VarId,
    pub fragments: Vec<Fragment<Expr<S>>>,
}

pub struct Constraints<S> {
    pub addr: Option<Expr<S>>,
}

pub type Expr<S> = crate::expr::Expr<SymbolId, S>;

pub type SymbolId = Symbol<BuiltinDefId, UserDefId>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Symbol<B, C> {
    Builtin(B),
    UserDef(C),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BuiltinDefId {
    Sizeof,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UserDefId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VarId(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub enum Fragment<E> {
    Byte(u8),
    Immediate(E, Width),
    LdInlineAddr(u8, E),
    Embedded(u8, E),
    Reloc(VarId),
    Reserved(E),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Width {
    Byte,
    Word,
}

pub struct SymbolTable<S>(pub Vec<Option<UserDef<Closure<S>, SectionId>>>);

#[derive(Clone, Debug, PartialEq)]
pub enum UserDef<F, S> {
    Closure(F),
    Section(S),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Closure<S> {
    pub expr: Expr<S>,
    pub location: VarId,
}

#[derive(Debug, PartialEq)]
pub struct SectionId(pub usize);

pub struct VarTable(pub Vec<Var>);

#[derive(Clone, Default)]
pub struct Var {
    pub value: Num,
}

impl<S> Data<S> {
    pub fn new() -> Self {
        Data {
            content: Content::new(),
            vars: VarTable::new(),
        }
    }
}

impl<S> Content<S> {
    pub fn new() -> Content<S> {
        Content {
            sections: Vec::new(),
            symbols: SymbolTable::new(),
        }
    }

    pub fn sections(&self) -> impl Iterator<Item = &Section<S>> {
        self.sections.iter()
    }

    pub fn add_section(&mut self, name: Option<UserDefId>, addr: VarId, size: VarId) {
        let section = SectionId(self.sections.len());
        self.sections.push(Section::new(addr, size));
        if let Some(name) = name {
            self.symbols.define(name, UserDef::Section(section))
        }
    }
}

impl<S> Section<S> {
    pub fn new(addr: VarId, size: VarId) -> Section<S> {
        Section {
            constraints: Constraints { addr: None },
            addr,
            size,
            fragments: Vec::new(),
        }
    }
}

impl<S> SymbolTable<S> {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn alloc(&mut self) -> UserDefId {
        let id = UserDefId(self.0.len());
        self.0.push(None);
        id
    }

    pub fn define(&mut self, UserDefId(id): UserDefId, def: UserDef<Closure<S>, SectionId>) {
        assert!(self.0[id].is_none());
        self.0[id] = Some(def);
    }

    pub fn get(&self, UserDefId(id): UserDefId) -> Option<&UserDef<Closure<S>, SectionId>> {
        self.0[id].as_ref()
    }
}

pub(super) struct LinkageContext<C, V> {
    pub content: C,
    pub vars: V,
    pub location: Num,
}

impl Var {
    pub fn refine(&mut self, value: Num) -> bool {
        let old_value = self.value.clone();
        let was_refined = match (old_value, &value) {
            (Num::Unknown, new_value) => *new_value != Num::Unknown,
            (
                Num::Range {
                    min: old_min,
                    max: old_max,
                },
                Num::Range {
                    min: new_min,
                    max: new_max,
                },
            ) => {
                assert!(*new_min >= old_min);
                assert!(*new_max <= old_max);
                *new_min > old_min || *new_max < old_max
            }
            (Num::Range { .. }, Num::Unknown) => {
                panic!("a symbol previously approximated is now unknown")
            }
        };
        self.value = value;
        was_refined
    }
}

impl VarTable {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn alloc(&mut self) -> VarId {
        let id = VarId(self.0.len());
        self.0.push(Default::default());
        id
    }
}

impl Index<VarId> for VarTable {
    type Output = Var;

    fn index(&self, VarId(id): VarId) -> &Self::Output {
        &self.0[id]
    }
}

impl IndexMut<VarId> for VarTable {
    fn index_mut(&mut self, VarId(id): VarId) -> &mut Self::Output {
        &mut self.0[id]
    }
}

impl From<SymbolId> for Atom<SymbolId> {
    fn from(id: SymbolId) -> Self {
        Atom::Name(id)
    }
}

#[cfg(test)]
impl From<UserDefId> for Atom<SymbolId> {
    fn from(id: UserDefId) -> Self {
        Atom::Name(id.into())
    }
}

impl From<SymbolId> for ExprOp<SymbolId> {
    fn from(id: SymbolId) -> Self {
        Atom::from(id).into()
    }
}

#[cfg(test)]
impl From<BuiltinDefId> for ExprOp<SymbolId> {
    fn from(builtin: BuiltinDefId) -> Self {
        Atom::from(Symbol::from(builtin)).into()
    }
}

#[cfg(test)]
impl From<UserDefId> for ExprOp<SymbolId> {
    fn from(id: UserDefId) -> Self {
        Atom::from(id).into()
    }
}

impl<B, C> Symbol<B, C> {
    pub fn content(self) -> Option<C> {
        match self {
            Symbol::Builtin(_) => None,
            Symbol::UserDef(id) => Some(id),
        }
    }
}

impl From<BuiltinDefId> for SymbolId {
    fn from(builtin: BuiltinDefId) -> Self {
        Symbol::Builtin(builtin)
    }
}

impl From<UserDefId> for SymbolId {
    fn from(id: UserDefId) -> Self {
        Symbol::UserDef(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_section_defines_name() {
        let mut program = Content::<()>::new();
        let name = program.symbols.alloc();
        program.add_section(Some(name), VarId(0), VarId(1));
        assert_eq!(
            program.symbols.get(name),
            Some(&UserDef::Section(SectionId(0)))
        )
    }
}
