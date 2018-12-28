use gbas::diag::*;
use std::fmt;
use std::fs::File;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() <= 1 {
        println!("{}: error: no input files", args[0]);
        std::process::exit(1)
    }
    let filename = &args[1];
    let mut diagnostics = |diagnostic| println!("{}", GbasDiagnostic(diagnostic));
    let mut config = gbas::Config {
        input: gbas::InputConfig::default(),
        diagnostics: gbas::DiagnosticsConfig::Output(&mut diagnostics),
    };
    if let Some(rom) = gbas::assemble(filename, &mut config) {
        let mut rom_file = File::create(filename.to_owned() + ".o").unwrap();
        rom_file.write_all(&rom.data).unwrap()
    }
}

struct GbasDiagnostic(Diagnostic);
struct GbasClause<'a>(&'a Clause);

impl fmt::Display for GbasDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for clause in &self.0.clauses {
            write!(f, "{}", GbasClause(clause))?
        }
        Ok(())
    }
}

impl<'a> fmt::Display for GbasClause<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.0.location {
            None => writeln!(f, "{}: {}: {}", self.0.file, self.0.tag, self.0.message),
            Some(location) => {
                let squiggle = location
                    .highlight
                    .as_ref()
                    .map_or_else(String::new, mk_squiggle);
                writeln!(
                    f,
                    "{}:{}: {}: {}\n{}{}",
                    self.0.file,
                    location.line,
                    self.0.tag,
                    self.0.message,
                    location.source,
                    squiggle,
                )
            }
        }
    }
}

fn mk_squiggle(range: &TextRange) -> String {
    assert_eq!(range.start.line, range.end.line);

    use std::cmp::max;
    let space_count = range.start.column_index;
    let tilde_count = max(range.end.column_index - space_count, 1);

    use std::iter::{once, repeat};
    let spaces = repeat(' ').take(space_count);
    let tildes = repeat('~').take(tilde_count);
    once('\n').chain(spaces).chain(tildes).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    static DUMMY_FILE: &str = "/my/file";

    #[test]
    fn render_elaborated_diagnostic() {
        let diagnostic = Diagnostic {
            clauses: vec![Clause {
                file: DUMMY_FILE.to_string(),
                tag: Tag::Error,
                message: "invocation of undefined macro `my_macro`".to_string(),
                location: Some(Location {
                    line: LineNumber(2),
                    source: "    my_macro a, $12".to_string(),
                    highlight: mk_highlight(LineNumber(2), 4, 12),
                }),
            }],
        };
        let expected = r"/my/file:2: error: invocation of undefined macro `my_macro`
    my_macro a, $12
    ~~~~~~~~
";
        assert_eq!(GbasDiagnostic(diagnostic).to_string(), expected)
    }

    #[test]
    fn render_diagnostic_without_source() {
        let diagnostic = Diagnostic {
            clauses: vec![Clause {
                file: DUMMY_FILE.to_string(),
                tag: Tag::Error,
                message: "file constains invalid UTF-8".to_string(),
                location: None,
            }],
        };
        let expected = r"/my/file: error: file constains invalid UTF-8
";
        assert_eq!(GbasDiagnostic(diagnostic).to_string(), expected);
    }

    #[test]
    fn highlight_eof_with_one_tilde() {
        let elaborated = Diagnostic {
            clauses: vec![Clause {
                file: DUMMY_FILE.to_string(),
                tag: Tag::Error,
                message: "unexpected end of file".into(),
                location: Some(Location {
                    line: LineNumber(2),
                    source: "dummy".to_string(),
                    highlight: mk_highlight(LineNumber(2), 5, 5),
                }),
            }],
        };
        let expected = r"/my/file:2: error: unexpected end of file
dummy
     ~
";
        assert_eq!(GbasDiagnostic(elaborated).to_string(), expected)
    }

    fn mk_highlight(line_number: LineNumber, start: usize, end: usize) -> Option<TextRange> {
        Some(TextRange {
            start: TextPosition {
                line: line_number.into(),
                column_index: start,
            },
            end: TextPosition {
                line: line_number.into(),
                column_index: end,
            },
        })
    }
}
