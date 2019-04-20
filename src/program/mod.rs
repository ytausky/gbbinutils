pub use self::builder::ProgramBuilder;

use crate::model::{Atom, Width};

mod builder;
mod link;
mod lowering;

type Immediate<S> = crate::model::Expr<Atom<NameId>, S>;

#[derive(Clone, Copy, Debug, PartialEq)]
struct RelocId(usize);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NameId(usize);

impl From<NameId> for Atom<NameId> {
    fn from(id: NameId) -> Self {
        Atom::Name(id)
    }
}

pub struct Program<S> {
    sections: Vec<Section<S>>,
    names: NameTable,
    relocs: usize,
}

struct Section<S> {
    constraints: Constraints<S>,
    addr: RelocId,
    size: RelocId,
    items: Vec<Node<S>>,
}

struct Constraints<S> {
    addr: Option<Immediate<S>>,
}

#[derive(Clone, Debug, PartialEq)]
enum Node<S> {
    Byte(u8),
    Expr(Immediate<S>, Width),
    LdInlineAddr(u8, Immediate<S>),
    Embedded(u8, Immediate<S>),
    Reserved(Immediate<S>),
    Symbol((NameId, S), Immediate<S>),
}

#[derive(Debug, PartialEq)]
enum NameDef {
    Reloc(RelocId),
    Section(SectionId),
}

#[derive(Debug, PartialEq)]
struct SectionId(usize);

impl<S> Program<S> {
    pub fn new() -> Program<S> {
        Program {
            sections: Vec::new(),
            names: NameTable::new(),
            relocs: 0,
        }
    }

    fn add_section(&mut self, name: Option<NameId>) {
        let section = SectionId(self.sections.len());
        let addr = self.alloc_reloc();
        let size = self.alloc_reloc();
        self.sections.push(Section::new(addr, size));
        if let Some(name) = name {
            self.names.define(name, NameDef::Section(section))
        }
    }

    fn alloc_reloc(&mut self) -> RelocId {
        let id = self.relocs;
        self.relocs += 1;
        RelocId(id)
    }
}

impl<S> Section<S> {
    pub fn new(addr: RelocId, size: RelocId) -> Section<S> {
        Section {
            constraints: Constraints { addr: None },
            addr,
            size,
            items: Vec::new(),
        }
    }
}

struct NameTable(Vec<Option<NameDef>>);

impl NameTable {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    fn alloc(&mut self) -> NameId {
        let id = NameId(self.0.len());
        self.0.push(None);
        id
    }

    fn define(&mut self, NameId(id): NameId, def: NameDef) {
        assert!(self.0[id].is_none());
        self.0[id] = Some(def);
    }

    fn get(&self, NameId(id): NameId) -> Option<&NameDef> {
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
        let name = program.names.alloc();
        program.add_section(Some(name));
        assert_eq!(
            program.names.get(name),
            Some(&NameDef::Section(SectionId(0)))
        )
    }
}
