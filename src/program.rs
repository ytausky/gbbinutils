pub struct Program {
    pub sections: Box<[Chunk]>,
}

pub struct Chunk {
    pub addr: usize,
    pub data: Box<[u8]>,
}

impl Program {
    pub fn into_rom(self) -> Box<[u8]> {
        let default = 0xffu8;
        let mut rom: Vec<u8> = Vec::new();
        for section in Vec::from(self.sections) {
            if !section.data.is_empty() {
                let end = section.addr + section.data.len();
                if rom.len() < end {
                    rom.resize(end, default)
                }
                rom[section.addr..end].copy_from_slice(&section.data)
            }
        }
        if rom.len() < MIN_ROM_LEN {
            rom.resize(MIN_ROM_LEN, default)
        }
        rom.into_boxed_slice()
    }
}

const MIN_ROM_LEN: usize = 0x8000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_object_converted_to_all_0xff_rom() {
        let object = Program {
            sections: Box::new([]),
        };
        let rom = object.into_rom();
        assert_eq!(*rom, [0xffu8; MIN_ROM_LEN][..])
    }

    #[test]
    fn section_placed_in_rom_starting_at_origin() {
        let byte = 0x42;
        let addr = 0x150;
        let object = Program {
            sections: Box::new([Chunk {
                addr,
                data: Box::new([byte]),
            }]),
        };
        let rom = object.into_rom();
        let mut expected = [0xffu8; MIN_ROM_LEN];
        expected[addr] = byte;
        assert_eq!(*rom, expected[..])
    }

    #[test]
    fn empty_section_does_not_extend_rom() {
        let addr = MIN_ROM_LEN + 1;
        let object = Program {
            sections: Box::new([Chunk {
                addr,
                data: Box::new([]),
            }]),
        };
        let rom = object.into_rom();
        assert_eq!(rom.len(), MIN_ROM_LEN)
    }
}
