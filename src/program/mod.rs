pub(crate) use self::builder::ProgramBuilder;

use self::link::VarTable;

use crate::model::{Atom, ExprOp, LocationCounter, Width};

mod builder;
mod link;
mod lowering;

pub struct LinkableProgram<S> {
    program: Program<S>,
    vars: VarTable,
}

impl<S> LinkableProgram<S> {
    pub fn new() -> Self {
        Self {
            program: Program::new(),
            vars: VarTable::new(),
        }
    }
}

type Expr<S> = crate::model::Expr<Atom<VarId, Symbol>, S>;
type Const<S> = crate::model::Expr<Atom<LocationCounter, Symbol>, S>;

impl<L> From<Symbol> for Atom<L, Symbol> {
    fn from(id: Symbol) -> Self {
        Atom::Name(id)
    }
}

#[cfg(test)]
impl<L> From<ProgramSymbol> for Atom<L, Symbol> {
    fn from(id: ProgramSymbol) -> Self {
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
impl<L> From<ProgramSymbol> for ExprOp<Atom<L, Symbol>> {
    fn from(id: ProgramSymbol) -> Self {
        Atom::from(id).into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct VarId(usize);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Symbol {
    Builtin(BuiltinSymbol),
    Program(ProgramSymbol),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BuiltinSymbol {
    Sizeof,
}

impl Symbol {
    fn program(self) -> Option<ProgramSymbol> {
        match self {
            Symbol::Builtin(_) => None,
            Symbol::Program(id) => Some(id),
        }
    }
}

impl From<BuiltinSymbol> for Symbol {
    fn from(builtin: BuiltinSymbol) -> Self {
        Symbol::Builtin(builtin)
    }
}

impl From<ProgramSymbol> for Symbol {
    fn from(id: ProgramSymbol) -> Self {
        Symbol::Program(id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProgramSymbol(usize);

pub struct Program<S> {
    sections: Vec<Section<S>>,
    symbols: SymbolTable<S>,
}

struct Section<S> {
    constraints: Constraints<S>,
    addr: VarId,
    size: VarId,
    items: Vec<Node<S>>,
}

struct Constraints<S> {
    addr: Option<Const<S>>,
}

#[derive(Clone, Debug, PartialEq)]
enum Node<S> {
    Byte(u8),
    Immediate(Const<S>, Width),
    LdInlineAddr(u8, Const<S>),
    Embedded(u8, Const<S>),
    Reloc(VarId),
    Reserved(Const<S>),
}

#[derive(Debug, PartialEq)]
enum ProgramDef<S> {
    Section(SectionId),
    Expr(Expr<S>),
}

#[derive(Debug, PartialEq)]
struct SectionId(usize);

impl<S> Program<S> {
    fn new() -> Program<S> {
        Program {
            sections: Vec::new(),
            symbols: SymbolTable::new(),
        }
    }

    fn add_section(&mut self, name: Option<ProgramSymbol>, addr: VarId, size: VarId) {
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

struct SymbolTable<S>(Vec<Option<ProgramDef<S>>>);

impl<S> SymbolTable<S> {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    fn alloc(&mut self) -> ProgramSymbol {
        let id = ProgramSymbol(self.0.len());
        self.0.push(None);
        id
    }

    fn define(&mut self, ProgramSymbol(id): ProgramSymbol, def: ProgramDef<S>) {
        assert!(self.0[id].is_none());
        self.0[id] = Some(def);
    }

    fn get(&self, ProgramSymbol(id): ProgramSymbol) -> Option<&ProgramDef<S>> {
        self.0[id].as_ref()
    }
}

pub struct BinaryObject {
    pub sections: Vec<BinarySection>,
}

impl BinaryObject {
    pub fn into_rom(self) -> Rom {
        let default = 0xffu8;
        let mut data: Vec<u8> = Vec::new();
        for section in self.sections {
            if !section.data.is_empty() {
                let end = section.addr + section.data.len();
                if data.len() < end {
                    data.resize(end, default)
                }
                data[section.addr..end].copy_from_slice(&section.data)
            }
        }
        if data.len() < MIN_ROM_LEN {
            data.resize(MIN_ROM_LEN, default)
        }
        Rom {
            data: data.into_boxed_slice(),
        }
    }
}

const MIN_ROM_LEN: usize = 0x8000;

pub struct Rom {
    pub data: Box<[u8]>,
}

pub struct BinarySection {
    pub addr: usize,
    pub data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_object_converted_to_all_0xff_rom() {
        let object = BinaryObject {
            sections: Vec::new(),
        };
        let rom = object.into_rom();
        assert_eq!(*rom.data, [0xffu8; MIN_ROM_LEN][..])
    }

    #[test]
    fn section_placed_in_rom_starting_at_origin() {
        let byte = 0x42;
        let addr = 0x150;
        let object = BinaryObject {
            sections: vec![BinarySection {
                addr,
                data: vec![byte],
            }],
        };
        let rom = object.into_rom();
        let mut expected = [0xffu8; MIN_ROM_LEN];
        expected[addr] = byte;
        assert_eq!(*rom.data, expected[..])
    }

    #[test]
    fn empty_section_does_not_extend_rom() {
        let addr = MIN_ROM_LEN + 1;
        let object = BinaryObject {
            sections: vec![BinarySection {
                addr,
                data: Vec::new(),
            }],
        };
        let rom = object.into_rom();
        assert_eq!(rom.data.len(), MIN_ROM_LEN)
    }

    #[test]
    fn add_section_defines_name() {
        let mut program = Program::<()>::new();
        let name = program.symbols.alloc();
        program.add_section(Some(name), VarId(0), VarId(1));
        assert_eq!(
            program.symbols.get(name),
            Some(&ProgramDef::Section(SectionId(0)))
        )
    }
}
