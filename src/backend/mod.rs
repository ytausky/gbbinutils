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

pub struct Rom;

impl Rom {
    pub fn new() -> Rom {
        Rom {}
    }
}

use ir;

impl ir::Object for Rom {
    fn add_label(&mut self, _label: &str) {}
    fn add_instruction(&mut self, _instruction: ir::Instruction) {}
}
