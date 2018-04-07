mod backend;
mod diagnostics;
mod frontend;
mod ir;

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

pub trait Backend {
    type Object: ir::Object;
    fn mk_object(&mut self) -> Self::Object;
}

struct DumpingBackend;

impl DumpingBackend {
    fn new() -> DumpingBackend {
        DumpingBackend {}
    }
}

impl Backend for DumpingBackend {
    type Object = ir::OutputDumper;
    fn mk_object(&mut self) -> Self::Object {
        ir::OutputDumper::new()
    }
}
