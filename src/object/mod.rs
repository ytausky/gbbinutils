use self::builder::Width;
use self::num::Num;

use crate::expr::{Atom, Expr, ExprOp, LocationCounter};

use std::ops::{Index, IndexMut};

pub mod builder;
pub mod num;

mod eval;

pub struct Object<S> {
    pub content: Content<S>,
    pub vars: VarTable,
}

pub struct Content<S> {
    sections: Vec<Section<S>>,
    symbols: SymbolTable<S>,
}

pub struct Section<S> {
    pub constraints: Constraints<S>,
    pub addr: VarId,
    pub size: VarId,
    pub items: Vec<Node<S>>,
}

pub struct Constraints<S> {
    pub addr: Option<Const<S>>,
}

pub type Const<S> = Expr<Atom<LocationCounter, SymbolId>, S>;

type SymbolId = Symbol<BuiltinId, ContentId>;

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
pub enum Node<S> {
    Byte(u8),
    Immediate(Const<S>, Width),
    LdInlineAddr(u8, Const<S>),
    Embedded(u8, Const<S>),
    Reloc(VarId),
    Reserved(Const<S>),
}

struct SymbolTable<S>(Vec<Option<ContentDef<ExprDef<S>, SectionId>>>);

#[derive(Clone, Debug, PartialEq)]
enum ContentDef<F, S> {
    Formula(F),
    Section(S),
}

#[derive(Clone, Debug, PartialEq)]
struct ExprDef<S> {
    expr: Formula<S>,
    location: VarId,
}

#[derive(Debug, PartialEq)]
pub struct SectionId(usize);

type Formula<S> = Expr<Atom<LocationCounter, SymbolId>, S>;

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
            items: Vec::new(),
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

    fn get(&self, ContentId(id): ContentId) -> Option<&ContentDef<ExprDef<S>, SectionId>> {
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

impl<L> From<SymbolId> for Atom<L, SymbolId> {
    fn from(id: SymbolId) -> Self {
        Atom::Name(id)
    }
}

#[cfg(test)]
impl<L> From<ContentId> for Atom<L, SymbolId> {
    fn from(id: ContentId) -> Self {
        Atom::Name(id.into())
    }
}

impl<L> From<SymbolId> for ExprOp<Atom<L, SymbolId>> {
    fn from(id: SymbolId) -> Self {
        Atom::from(id).into()
    }
}

#[cfg(test)]
impl<L> From<BuiltinId> for ExprOp<Atom<L, SymbolId>> {
    fn from(builtin: BuiltinId) -> Self {
        Atom::from(Symbol::from(builtin)).into()
    }
}

#[cfg(test)]
impl<L> From<ContentId> for ExprOp<Atom<L, SymbolId>> {
    fn from(id: ContentId) -> Self {
        Atom::from(id).into()
    }
}

impl<B, C> Symbol<B, C> {
    fn content(self) -> Option<C> {
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
