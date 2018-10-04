use std::{fs::File, io::Write};

mod backend;
mod codebase;
mod diagnostics;
mod frontend;
mod instruction;
mod span;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Width {
    Byte,
    Word,
}

impl Width {
    fn len(self) -> i32 {
        match self {
            Width::Byte => 1,
            Width::Word => 2,
        }
    }
}

pub use diagnostics::TerminalOutput;

pub struct DiagnosticsConfig<'a> {
    pub output: &'a mut dyn diagnostics::DiagnosticsOutput,
}

pub fn assemble_rom<'a>(name: &str, config: DiagnosticsConfig<'a>) {
    let codebase = codebase::FileCodebase::new(codebase::StdFileSystem::new());
    let diagnostics = diagnostics::OutputForwarder {
        output: config.output,
        codebase: &codebase.cache,
    };
    let object = frontend::analyze_file(
        name.to_string(),
        &codebase,
        span::SimpleTokenTracker {},
        backend::ObjectBuilder::new(),
        &diagnostics,
    );
    let rom = backend::link(object, &diagnostics).into_rom();
    let mut rom_file = File::create(name.to_owned() + ".o").unwrap();
    rom_file.write_all(&rom.data).unwrap()
}
