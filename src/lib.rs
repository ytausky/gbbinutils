pub use backend::Rom;
pub use codebase::StdFileSystem;
pub use diagnostics::TerminalOutput;

mod backend;
mod codebase;
mod diagnostics;
mod expr;
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

pub struct Config<'a> {
    pub input: &'a mut dyn codebase::FileSystem,
    pub output: &'a mut dyn diagnostics::DiagnosticsOutput,
}

pub fn assemble<'a>(name: &str, config: &mut Config<'a>) -> Rom {
    let codebase = codebase::FileCodebase::new(config.input);
    let mut diagnostics = diagnostics::OutputForwarder {
        output: config.output,
        codebase: &codebase.cache,
    };
    match frontend::analyze_file(
        name.to_string(),
        &codebase,
        span::SimpleTokenTracker {},
        backend::ObjectBuilder::new(),
        &mut diagnostics,
    ) {
        Ok(object) => backend::link(object, &mut diagnostics).into_rom(),
        Err(_) => unimplemented!(),
    }
}
