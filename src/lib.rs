use std::fmt::Debug;

mod backend;
mod codebase;
mod diagnostics;
mod frontend;

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
    frontend::analyze_file(
        name.to_string(),
        &codebase,
        diagnostics::SimpleTokenTracker {},
        backend::ObjectBuilder::new(&diagnostics),
        &diagnostics,
    );
}

struct OutputDumper;

impl OutputDumper {
    pub fn new() -> OutputDumper {
        OutputDumper {}
    }
}

impl<R: Debug> backend::Backend<R> for OutputDumper {
    type Section = Self;
    fn mk_section(&mut self) -> Self::Section {
        OutputDumper {}
    }
}

impl<R: Debug> backend::Section<R> for OutputDumper {
    fn add_label(&mut self, (label, _): (&str, R)) {
        println!("Define symbol: {}", label)
    }

    fn emit_item(&mut self, item: backend::Item<R>) {
        println!("Emit {:?}", item)
    }
}
