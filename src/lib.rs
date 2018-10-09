pub use backend::Rom;
pub use diagnostics::TerminalOutput;

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

pub struct DiagnosticsConfig<'a> {
    pub output: &'a mut dyn diagnostics::DiagnosticsOutput,
}

pub fn assemble<'a>(name: &str, config: &mut DiagnosticsConfig<'a>) -> Rom {
    let codebase = codebase::FileCodebase::new(codebase::StdFileSystem::new());
    let mut diagnostics = diagnostics::OutputForwarder {
        output: config.output,
        codebase: &codebase.cache,
    };
    let object = frontend::analyze_file(
        name.to_string(),
        &codebase,
        span::SimpleTokenTracker {},
        backend::ObjectBuilder::new(),
        &mut diagnostics,
    );
    backend::link(object, &mut diagnostics).into_rom()
}
