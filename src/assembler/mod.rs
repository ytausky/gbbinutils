use self::session::{Interner, ReentrancyActions, Session};

use crate::codebase::{CodebaseError, FileCodebase, FileSystem, StdFileSystem};
use crate::diagnostics::{mk_diagnostic, Diagnostic, OutputForwarder};
use crate::link::Program;
use crate::{Config, DiagnosticsConfig, InputConfig};

mod keywords;
mod semantics;
pub mod session;
mod syntax;

pub struct Assembler<'a> {
    config: &'a mut Config<'a>,
}

impl<'a> Assembler<'a> {
    pub fn new(config: &'a mut Config<'a>) -> Self {
        Self { config }
    }

    /// Parses input files and generates a ROM.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut config = gbbinutils::Config::default();
    /// let mut assembler = gbbinutils::assembler::Assembler::new(&mut config);
    /// assembler.assemble("game.s");
    /// ```
    pub fn assemble(&mut self, name: &str) -> Option<Program> {
        let mut input_holder = None;
        let mut diagnostics_holder = None;
        let input: &mut dyn FileSystem = match &mut self.config.input {
            InputConfig::Default => input_holder.get_or_insert_with(StdFileSystem::new),
            InputConfig::Custom(input) => *input,
        };
        let diagnostics: &mut dyn FnMut(Diagnostic) = match &mut self.config.diagnostics {
            DiagnosticsConfig::Ignore => diagnostics_holder.get_or_insert(|_| {}),
            DiagnosticsConfig::Output(diagnostics) => *diagnostics,
        };
        try_assemble(name, input, diagnostics)
            .map_err(|error| diagnostics(mk_diagnostic(name, &error.into())))
            .ok()
    }
}

fn try_assemble<'a>(
    name: &str,
    input: &'a mut dyn FileSystem,
    output: &'a mut dyn FnMut(Diagnostic),
) -> Result<Program, CodebaseError> {
    let codebase = FileCodebase::new(input);
    let diagnostics = OutputForwarder { output };
    let mut session = Session::new(codebase, diagnostics);
    let name = session.intern(name);
    session.analyze_file(name, None)?;

    let result = Program::link(
        session.builder.object,
        session.codebase,
        session.diagnostics,
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::codebase::fake::MockFileSystem;
    use crate::diagnostics::{Clause, Tag};

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
        let mut output = |diagnostic| diagnostics.push(diagnostic);
        let mut config = Config {
            input: InputConfig::Custom(fs),
            diagnostics: DiagnosticsConfig::Output(&mut output),
        };
        let mut assembler = Assembler::new(&mut config);
        assembler.assemble(path);
        diagnostics
    }
}
