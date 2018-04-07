mod backend;
mod diagnostics;
mod frontend;
mod ir;

pub fn analyze_file(name: &str) {
    frontend::analyze_file(name.to_string(), DumpingBackend::new());
}

pub fn assemble_rom(name: &str) {
    let _rom = frontend::analyze_file(name.to_string(), backend::RomGenerator::new());
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
