mod codegen;

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
