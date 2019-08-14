pub(crate) use self::builder::ProgramBuilder;

use self::num::Num;

use crate::model::{Atom, ExprOp, LocationCounter, Width};

pub mod num;

mod builder;
mod eval;
mod lowering;

pub struct Object<S> {
    pub program: Content<S>,
    pub vars: VarTable,
}

impl<S> Object<S> {
    pub fn new() -> Self {
        Object {
            program: Content::new(),
            vars: VarTable::new(),
        }
    }
}

pub(super) struct LinkageContext<P, V> {
    pub program: P,
    pub vars: V,
    pub location: Num,
}

pub struct VarTable(pub Vec<Var>);

#[derive(Clone, Default)]
pub struct Var {
    pub value: Num,
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

type Expr<S> = crate::model::Expr<Atom<VarId, Symbol>, S>;
pub type Const<S> = crate::model::Expr<Atom<LocationCounter, Symbol>, S>;

impl<L> From<Symbol> for Atom<L, Symbol> {
    fn from(id: Symbol) -> Self {
        Atom::Name(id)
    }
}

#[cfg(test)]
impl<L> From<ContentSymbol> for Atom<L, Symbol> {
    fn from(id: ContentSymbol) -> Self {
        Atom::Name(id.into())
    }
}

impl<L> From<Symbol> for ExprOp<Atom<L, Symbol>> {
    fn from(id: Symbol) -> Self {
        Atom::from(id).into()
    }
}

#[cfg(test)]
impl<L> From<BuiltinSymbol> for ExprOp<Atom<L, Symbol>> {
    fn from(builtin: BuiltinSymbol) -> Self {
        Atom::from(Symbol::from(builtin)).into()
    }
}

#[cfg(test)]
impl<L> From<ContentSymbol> for ExprOp<Atom<L, Symbol>> {
    fn from(id: ContentSymbol) -> Self {
        Atom::from(id).into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VarId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Symbol {
    Builtin(BuiltinSymbol),
    Content(ContentSymbol),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BuiltinSymbol {
    Sizeof,
}

impl Symbol {
    fn content(self) -> Option<ContentSymbol> {
        match self {
            Symbol::Builtin(_) => None,
            Symbol::Content(id) => Some(id),
        }
    }
}

impl From<BuiltinSymbol> for Symbol {
    fn from(builtin: BuiltinSymbol) -> Self {
        Symbol::Builtin(builtin)
    }
}

impl From<ContentSymbol> for Symbol {
    fn from(id: ContentSymbol) -> Self {
        Symbol::Content(id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContentSymbol(pub usize);

pub struct Content<S> {
    pub sections: Vec<Section<S>>,
    pub symbols: SymbolTable<S>,
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

#[derive(Clone, Debug, PartialEq)]
pub enum Node<S> {
    Byte(u8),
    Immediate(Const<S>, Width),
    LdInlineAddr(u8, Const<S>),
    Embedded(u8, Const<S>),
    Reloc(VarId),
    Reserved(Const<S>),
}

#[derive(Debug, PartialEq)]
pub enum ProgramDef<S> {
    Section(SectionId),
    Expr(Expr<S>),
}

#[derive(Debug, PartialEq)]
pub struct SectionId(pub usize);

impl<S> Content<S> {
    pub fn new() -> Content<S> {
        Content {
            sections: Vec::new(),
            symbols: SymbolTable::new(),
        }
    }

    pub fn add_section(&mut self, name: Option<ContentSymbol>, addr: VarId, size: VarId) {
        let section = SectionId(self.sections.len());
        self.sections.push(Section::new(addr, size));
        if let Some(name) = name {
            self.symbols.define(name, ProgramDef::Section(section))
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

pub struct SymbolTable<S>(pub Vec<Option<ProgramDef<S>>>);

impl<S> SymbolTable<S> {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn alloc(&mut self) -> ContentSymbol {
        let id = ContentSymbol(self.0.len());
        self.0.push(None);
        id
    }

    pub fn define(&mut self, ContentSymbol(id): ContentSymbol, def: ProgramDef<S>) {
        assert!(self.0[id].is_none());
        self.0[id] = Some(def);
    }

    fn get(&self, ContentSymbol(id): ContentSymbol) -> Option<&ProgramDef<S>> {
        self.0[id].as_ref()
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
            Some(&ProgramDef::Section(SectionId(0)))
        )
    }
}
