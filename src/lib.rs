pub use backend::Rom;
pub use codebase::StdFileSystem;
pub use diagnostics::TerminalOutput;

use codebase::CodebaseError;
use diagnostics::{mk_diagnostic, Message};

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

pub fn assemble<'a>(name: &str, config: &mut Config<'a>) -> Option<Rom> {
    let result = try_assemble(name, config);
    match result {
        Ok(rom) => Some(rom),
        Err(CodebaseError::IoError(_)) => unimplemented!(),
        Err(CodebaseError::Utf8Error) => {
            config
                .output
                .emit(mk_diagnostic(name, &Message::InvalidUtf8));
            None
        }
    }
}

fn try_assemble<'a>(name: &str, config: &mut Config<'a>) -> Result<Rom, CodebaseError> {
    let codebase = codebase::FileCodebase::new(config.input);
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
    )?;
    Ok(backend::link(object, &mut diagnostics).into_rom())
}

#[cfg(test)]
mod tests {
    use super::*;
    use diagnostics::Diagnostic;
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
            Ok(self.files.get(filename).unwrap().clone())
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
                input: &mut fs,
                output: &mut output,
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
}
