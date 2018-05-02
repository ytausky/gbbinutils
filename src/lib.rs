mod backend;
mod codebase;
mod diagnostics;
mod frontend;

pub fn analyze_file(name: &str) {
    let codebase = codebase::FileCodebase::new(codebase::StdFileSystem::new());
    frontend::analyze_file(name.to_string(), codebase, OutputDumper::new());
}

use std::fs::File;
use std::io::Write;

pub fn assemble_rom(name: &str) {
    let codebase = codebase::FileCodebase::new(codebase::StdFileSystem::new());
    let diagnostics = diagnostics::DiagnosticsDumper::new();
    let rom = frontend::analyze_file(name.to_string(), codebase, backend::Rom::new(&diagnostics));
    let mut file = File::create("my_rom.gb").unwrap();
    file.write_all(rom.as_slice()).unwrap();
}

struct OutputDumper;

impl OutputDumper {
    pub fn new() -> OutputDumper {
        OutputDumper {}
    }
}

impl<R> backend::Backend<R> for OutputDumper {
    fn add_label(&mut self, (label, _): (&str, R)) {
        println!("Define symbol: {}", label)
    }

    fn emit_item(&mut self, item: backend::Item) {
        println!("Emit {:?}", item)
    }
}
