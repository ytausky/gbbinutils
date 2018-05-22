use std::{fmt::Debug, fs::File, io::Write};

mod backend;
mod codebase;
mod diagnostics;
mod frontend;
mod session;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Width {
    Byte,
    Word,
}

impl Width {
    fn len(&self) -> usize {
        match self {
            Width::Byte => 1,
            Width::Word => 2,
        }
    }
}

pub fn analyze_file(name: &str) {
    let codebase = codebase::FileCodebase::new(codebase::StdFileSystem::new());
    let diagnostics = diagnostics::TerminalDiagnostics::new(&codebase.cache);
    frontend::analyze_file(
        name.to_string(),
        &codebase,
        diagnostics::SimpleTokenTracker {},
        OutputDumper::new(),
        &diagnostics,
    );
}

pub fn assemble_rom(name: &str) {
    let codebase = codebase::FileCodebase::new(codebase::StdFileSystem::new());
    let diagnostics = diagnostics::TerminalDiagnostics::new(&codebase.cache);
    let rom = frontend::analyze_file(
        name.to_string(),
        &codebase,
        diagnostics::SimpleTokenTracker {},
        backend::ObjectBuilder::new(&diagnostics),
        &diagnostics,
    ).into_rom();
    let mut rom_file = File::open(name.to_owned() + ".o").unwrap();
    rom_file.write_all(&rom.data).unwrap()
}

struct OutputDumper;

impl OutputDumper {
    pub fn new() -> OutputDumper {
        OutputDumper {}
    }
}

impl<R: Debug> backend::Backend<R> for OutputDumper {
    type Object = ();

    fn add_label(&mut self, (label, _): (impl Into<String>, R)) {
        println!("Define symbol: {}", label.into())
    }

    fn emit_item(&mut self, item: backend::Item<R>) {
        println!("Emit {:?}", item)
    }

    fn into_object(self) {}
}
