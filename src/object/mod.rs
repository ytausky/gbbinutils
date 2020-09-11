use self::num::Num;

use crate::expr::{Atom, ExprOp};
use crate::session::builder::Width;

use std::ops::{Index, IndexMut};

pub mod eval;
pub mod num;

pub struct Object<S> {
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
    pub fragments: Vec<Fragment<S>>,
}

pub struct Constraints<S> {
    pub addr: Option<Expr<S>>,
}

pub type Expr<S> = crate::expr::Expr<SymbolId, S>;

pub type SymbolId = Symbol<BuiltinId, ContentId>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Symbol<B, C> {
    Builtin(B),
    Content(C),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BuiltinId {
    Sizeof,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContentId(usize);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VarId(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub enum Fragment<S> {
    Byte(u8),
    Immediate(Expr<S>, Width),
    LdInlineAddr(u8, Expr<S>),
    Embedded(u8, Expr<S>),
    Reloc(VarId),
    Reserved(Expr<S>),
}

pub struct SymbolTable<S>(Vec<Option<ContentDef<ExprDef<S>, SectionId>>>);

#[derive(Clone, Debug, PartialEq)]
pub enum ContentDef<F, S> {
    Expr(F),
    Section(S),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExprDef<S> {
    pub expr: Expr<S>,
    pub location: VarId,
}

#[derive(Debug, PartialEq)]
pub struct SectionId(pub usize);

pub struct VarTable(Vec<Var>);

#[derive(Clone, Default)]
pub struct Var {
    pub value: Num,
}

impl<S> Object<S> {
    pub fn new() -> Self {
        Object {
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

    pub fn add_section(&mut self, name: Option<ContentId>, addr: VarId, size: VarId) {
        let section = SectionId(self.sections.len());
        self.sections.push(Section::new(addr, size));
        if let Some(name) = name {
            self.symbols.define(name, ContentDef::Section(section))
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

    pub fn alloc(&mut self) -> ContentId {
        let id = ContentId(self.0.len());
        self.0.push(None);
        id
    }

    pub fn define(&mut self, ContentId(id): ContentId, def: ContentDef<ExprDef<S>, SectionId>) {
        assert!(self.0[id].is_none());
        self.0[id] = Some(def);
    }

    pub fn get(&self, ContentId(id): ContentId) -> Option<&ContentDef<ExprDef<S>, SectionId>> {
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
impl From<ContentId> for Atom<SymbolId> {
    fn from(id: ContentId) -> Self {
        Atom::Name(id.into())
    }
}

impl From<SymbolId> for ExprOp<SymbolId> {
    fn from(id: SymbolId) -> Self {
        Atom::from(id).into()
    }
}

#[cfg(test)]
impl From<BuiltinId> for ExprOp<SymbolId> {
    fn from(builtin: BuiltinId) -> Self {
        Atom::from(Symbol::from(builtin)).into()
    }
}

#[cfg(test)]
impl From<ContentId> for ExprOp<SymbolId> {
    fn from(id: ContentId) -> Self {
        Atom::from(id).into()
    }
}

impl<B, C> Symbol<B, C> {
    pub fn content(self) -> Option<C> {
        match self {
            Symbol::Builtin(_) => None,
            Symbol::Content(id) => Some(id),
        }
    }
}

impl From<BuiltinId> for SymbolId {
    fn from(builtin: BuiltinId) -> Self {
        Symbol::Builtin(builtin)
    }
}

impl From<ContentId> for SymbolId {
    fn from(id: ContentId) -> Self {
        Symbol::Content(id)
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
            Some(&ContentDef::Section(SectionId(0)))
        )
    }
}
