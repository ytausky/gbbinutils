mod backend;
mod codebase;
mod diagnostics;
mod frontend;

pub fn analyze_file(name: &str) {
    frontend::analyze_file(name.to_string(), OutputDumper::new());
}

use std::fs::File;
use std::io::Write;

pub fn assemble_rom(name: &str) {
    let diagnostics = DiagnosticsDumper::new();
    let rom = frontend::analyze_file(name.to_string(), backend::Rom::new(&diagnostics));
    let mut file = File::create("my_rom.gb").unwrap();
    file.write_all(rom.as_slice()).unwrap();
}

struct DiagnosticsDumper;

impl DiagnosticsDumper {
    pub fn new() -> DiagnosticsDumper {
        DiagnosticsDumper {}
    }
}

impl diagnostics::DiagnosticsListener<()> for DiagnosticsDumper {
    fn emit_diagnostic(&self, diagnostic: diagnostics::Diagnostic<()>) {
        println!("{:?}", diagnostic)
    }
}

struct OutputDumper;

impl OutputDumper {
    pub fn new() -> OutputDumper {
        OutputDumper {}
    }
}

impl backend::Backend<()> for OutputDumper {
    fn add_label(&mut self, (label, _): (&str, ())) {
        println!("Define symbol: {}", label)
    }

    fn emit_item(&mut self, item: backend::Item) {
        println!("Emit {:?}", item)
    }
}
