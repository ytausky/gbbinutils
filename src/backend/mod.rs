mod codegen;

pub struct RomGenerator;

impl RomGenerator {
    pub fn new() -> RomGenerator {
        RomGenerator {}
    }
}

use Backend;

impl Backend for RomGenerator {
    type Object = Rom;
    fn mk_object(&mut self) -> Self::Object {
        Rom::new()
    }
}

pub struct Rom {
    data: Vec<u8>,
    counter: usize,
}

impl Rom {
    pub fn new() -> Rom {
        Rom {
            data: vec![0x00; 0x8000],
            counter: 0x0000,
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        self.data.as_slice()
    }
}

use ir;

impl ir::Object for Rom {
    fn add_label(&mut self, _label: &str) {}
    fn add_instruction(&mut self, _instruction: ir::Instruction) {}

    fn emit_byte(&mut self, byte: u8) {
        self.data[self.counter] = byte;
        self.counter +=1;
    }
}
