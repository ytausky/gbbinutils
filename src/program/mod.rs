pub use self::builder::ProgramBuilder;

use crate::model::Width;

mod builder;
mod link;
mod lowering;

type RelocExpr<S> = crate::model::RelocExpr<NameId, S>;

#[derive(Clone, Copy)]
struct ValueId(usize);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NameId(usize);

pub struct Program<S> {
    sections: Vec<Section<S>>,
    names: NameTable,
    relocs: usize,
}

struct Section<S> {
    name: Option<String>,
    addr: Option<RelocExpr<S>>,
    size: ValueId,
    items: Vec<Node<S>>,
}

#[derive(Clone, Debug, PartialEq)]
enum Node<S> {
    Byte(u8),
    Expr(RelocExpr<S>, Width),
    LdInlineAddr(u8, RelocExpr<S>),
    Embedded(u8, RelocExpr<S>),
    Symbol((NameId, S), RelocExpr<S>),
}

enum NameDef {
    Value(ValueId),
}

impl<S> Program<S> {
    pub fn new() -> Program<S> {
        Program {
            sections: Vec::new(),
            names: NameTable::new(),
            relocs: 0,
        }
    }

    fn add_section(&mut self, name: Option<String>) {
        let size = self.alloc_reloc();
        self.sections.push(Section::new(name, size))
    }

    fn alloc_reloc(&mut self) -> ValueId {
        let id = self.relocs;
        self.relocs += 1;
        ValueId(id)
    }
}

impl<S> Section<S> {
    pub fn new(name: Option<String>, size: ValueId) -> Section<S> {
        Section {
            name,
            addr: None,
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

    fn alloc_name(&mut self) -> NameId {
        let id = NameId(self.0.len());
        self.0.push(None);
        id
    }

    fn define_name(&mut self, NameId(id): NameId, def: NameDef) {
        assert!(self.0[id].is_none());
        self.0[id] = Some(def);
    }

    fn get_name_def(&self, NameId(id): NameId) -> Option<&NameDef> {
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
    pub name: Option<Box<str>>,
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
                name: None,
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
                name: None,
                addr,
                data: Vec::new(),
            }],
        };
        let rom = object.into_rom();
        assert_eq!(rom.data.len(), MIN_ROM_LEN)
    }

    #[test]
    fn new_section_has_name() {
        let name = "my_section";
        let mut program = Program::<()>::new();
        program.add_section(Some(name.into()));
        assert_eq!(program.sections[0].name, Some(name.into()))
    }
}
