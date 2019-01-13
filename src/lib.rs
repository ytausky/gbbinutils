//! An assembler for Game Boy.
//!
//! `gbas` is an assembler targeting Game Boy, Game Boy Pocket, Game Boy Light, and Game Boy Color.
//! Its customizable IO functions make it suitable for embedding in other tools, in unit tests, etc.

pub use crate::backend::Rom;

use crate::codebase::{CodebaseError, StdFileSystem};
use crate::diag::*;
use crate::frontend::Assemble;

mod backend;
mod codebase;
pub mod diag;
mod expr;
mod frontend;
mod instruction;
mod program;

#[derive(Default)]
pub struct Config<'a> {
    pub input: InputConfig<'a>,
    pub diagnostics: DiagnosticsConfig<'a>,
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

pub enum DiagnosticsConfig<'a> {
    Ignore,
    Output(&'a mut dyn FnMut(diag::Diagnostic)),
}

impl<'a> Default for DiagnosticsConfig<'a> {
    fn default() -> Self {
        DiagnosticsConfig::Ignore
    }
}

/// Parses input files and generates a ROM.
///
/// # Examples
///
/// ```rust
/// let rom = gbas::assemble("game.s", &mut gbas::Config::default());
/// assert!(rom.is_none())
/// ```
pub fn assemble(name: &str, config: &mut Config) -> Option<Rom> {
    let mut input_holder = None;
    let mut diagnostics_holder = None;
    let input: &mut dyn codebase::FileSystem = match config.input {
        InputConfig::Default => input_holder.get_or_insert_with(StdFileSystem::new),
        InputConfig::Custom(ref mut input) => *input,
    };
    let diagnostics: &mut dyn FnMut(diag::Diagnostic) = match config.diagnostics {
        DiagnosticsConfig::Ignore => diagnostics_holder.get_or_insert(|_| {}),
        DiagnosticsConfig::Output(ref mut diagnostics) => *diagnostics,
    };
    try_assemble(name, input, diagnostics)
        .map_err(|error| diagnostics(mk_diagnostic(name, &error.into())))
        .ok()
}

fn try_assemble(
    name: &str,
    input: &mut dyn codebase::FileSystem,
    output: &mut dyn FnMut(diag::Diagnostic),
) -> Result<Rom, CodebaseError> {
    let codebase = codebase::FileCodebase::new(input);
    let diagnostics = &mut DiagnosticsSystem::new(&codebase.cache, output);
    let mut builder = program::ProgramBuilder::new();
    builder.assemble(name, &codebase, diagnostics)?;
    let object = builder.into_object();
    Ok(object.link(diagnostics).into_rom())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag::Diagnostic;
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

    #[test]
    fn invalid_utf8() {
        let path = "/my/file";
        let mut fs = MockFileSystem::new();
        fs.add(path, &[0x5a, 0x0a, 0xf6, 0xa6]);
        assert_eq!(
            collect_diagnostics(path, &mut fs),
            [Diagnostic {
                clauses: vec![Clause {
                    file: path.to_string(),
                    tag: Tag::Error,
                    message: "file contains invalid UTF-8".to_string(),
                    excerpt: None
                }]
            }]
        )
    }

    #[test]
    fn nonexistent_file() {
        let path = "/my/file";
        assert_eq!(
            collect_diagnostics(path, &mut MockFileSystem::new()),
            [Diagnostic {
                clauses: vec![Clause {
                    file: path.to_string(),
                    tag: Tag::Error,
                    message: "file does not exist".to_string(),
                    excerpt: None
                }]
            }]
        )
    }

    fn collect_diagnostics(path: &str, fs: &mut MockFileSystem) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        assemble(
            path,
            &mut Config {
                input: InputConfig::Custom(fs),
                diagnostics: DiagnosticsConfig::Output(&mut |diagnostic| {
                    diagnostics.push(diagnostic)
                }),
            },
        );
        diagnostics
    }
}
