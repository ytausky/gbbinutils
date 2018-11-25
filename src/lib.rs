pub use crate::backend::Rom;

use crate::codebase::{CodebaseError, StdFileSystem};
use crate::diagnostics::{mk_diagnostic, TerminalOutput};

mod backend;
mod codebase;
mod diagnostics;
mod expr;
mod frontend;
mod instruction;
mod span;

#[derive(Default)]
pub struct Config<'a> {
    pub input: InputConfig<'a>,
    pub output: OutputConfig<'a>,
}

pub enum InputConfig<'a> {
    Default,
    Custom(&'a mut dyn codebase::FileSystem),
}

impl<'a> Default for InputConfig<'a> {
    fn default() -> Self {
        InputConfig::Default
    }
}

pub enum OutputConfig<'a> {
    Default,
    Custom(&'a mut dyn diagnostics::DiagnosticsOutput),
}

impl<'a> Default for OutputConfig<'a> {
    fn default() -> Self {
        OutputConfig::Default
    }
}

pub fn assemble<'a>(name: &str, config: &mut Config<'a>) -> Option<Rom> {
    let mut output_holder = None;
    let output: &mut dyn diagnostics::DiagnosticsOutput = match config.output {
        OutputConfig::Default => output_holder.get_or_insert_with(|| TerminalOutput {}),
        OutputConfig::Custom(ref mut output) => *output,
    };
    let result = {
        let mut input_holder = None;
        let input: &mut dyn codebase::FileSystem = match config.input {
            InputConfig::Default => input_holder.get_or_insert_with(|| StdFileSystem::new()),
            InputConfig::Custom(ref mut input) => *input,
        };
        try_assemble(name, input, output)
    };
    match result {
        Ok(rom) => Some(rom),
        Err(error) => {
            output.emit(mk_diagnostic(name, &error.into()));
            None
        }
    }
}

fn try_assemble<'a>(
    name: &str,
    input: &mut dyn codebase::FileSystem,
    output: &mut dyn diagnostics::DiagnosticsOutput,
) -> Result<Rom, CodebaseError> {
    let codebase = codebase::FileCodebase::new(input);
    let mut diagnostics = diagnostics::OutputForwarder {
        output,
        codebase: &codebase.cache,
    };
    let object = frontend::analyze_file(
        name.to_string(),
        &codebase,
        span::SimpleTokenTracker {},
        backend::ObjectBuilder::new(),
        &mut diagnostics,
    )?;
    Ok(backend::link(object, &mut diagnostics).into_rom())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::Diagnostic;
    use std::collections::HashMap;
    use std::io;

    struct MockFileSystem {
        files: HashMap<String, Vec<u8>>,
    }

    impl MockFileSystem {
        fn new() -> MockFileSystem {
            MockFileSystem {
                files: HashMap::new(),
            }
        }

        fn add(&mut self, name: impl Into<String>, data: &[u8]) {
            self.files.insert(name.into(), data.into());
        }
    }

    impl codebase::FileSystem for MockFileSystem {
        fn read_file(&self, filename: &str) -> io::Result<Vec<u8>> {
            self.files
                .get(filename)
                .cloned()
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "file does not exist"))
        }
    }

    struct MockDiagnosticOutput {
        diagnostics: Vec<Diagnostic<String>>,
    }

    impl MockDiagnosticOutput {
        fn new() -> MockDiagnosticOutput {
            MockDiagnosticOutput {
                diagnostics: Vec::new(),
            }
        }
    }

    impl diagnostics::DiagnosticsOutput for MockDiagnosticOutput {
        fn emit(&mut self, diagnostic: Diagnostic<String>) {
            self.diagnostics.push(diagnostic)
        }
    }

    #[test]
    fn invalid_utf8() {
        let path = "/my/file";
        let mut fs = MockFileSystem::new();
        fs.add(path, &[0x5a, 0x0a, 0xf6, 0xa6]);
        let mut output = MockDiagnosticOutput::new();
        {
            let mut config = Config {
                input: InputConfig::Custom(&mut fs),
                output: OutputConfig::Custom(&mut output),
            };
            assemble(path, &mut config);
        }
        assert_eq!(
            output.diagnostics,
            [Diagnostic {
                file: path.to_string(),
                message: "file contains invalid UTF-8".to_string(),
                location: None
            }]
        )
    }

    #[test]
    fn nonexistent_file() {
        let path = "/my/file";
        let mut fs = MockFileSystem::new();
        let mut output = MockDiagnosticOutput::new();
        {
            let mut config = Config {
                input: InputConfig::Custom(&mut fs),
                output: OutputConfig::Custom(&mut output),
            };
            assemble(path, &mut config);
        }
        assert_eq!(
            output.diagnostics,
            [Diagnostic {
                file: path.to_string(),
                message: "file does not exist".to_string(),
                location: None
            }]
        )
    }
}
