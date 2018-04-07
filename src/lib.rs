mod backend;
mod diagnostics;
mod frontend;

pub fn analyze_file(name: &str) {
    frontend::analyze_file(name.to_string(), DumpingBackend::new());
}

use std::fs::File;
use std::io::Write;

pub fn assemble_rom(name: &str) {
    let rom = frontend::analyze_file(name.to_string(), backend::RomGenerator::new());
    let mut file = File::create("my_rom.gb").unwrap();
    file.write_all(rom.as_slice()).unwrap();
}

struct DumpingBackend;

impl DumpingBackend {
    fn new() -> DumpingBackend {
        DumpingBackend {}
    }
}

impl backend::Backend for DumpingBackend {
    type Object = OutputDumper;
    fn mk_object(&mut self) -> Self::Object {
        OutputDumper::new()
    }
}

struct OutputDumper;

impl OutputDumper {
    pub fn new() -> OutputDumper {
        OutputDumper {}
    }
}

impl backend::Object for OutputDumper {
    fn add_instruction(&mut self, instruction: backend::Instruction) {
        println!("{:?}", instruction)
    }

    fn add_label(&mut self, label: &str) {
        println!("Define symbol: {}", label)
    }

    fn emit_byte(&mut self, byte: u8) {
        println!("Emit byte: {:x}", byte)
    }
}
