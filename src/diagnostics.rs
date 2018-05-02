pub trait DiagnosticsListener<R> {
    fn emit_diagnostic(&self, diagnostic: Diagnostic<R>);
}

#[derive(Debug, PartialEq)]
pub enum Diagnostic<R> {
    OperandCount { actual: usize, expected: usize },
    UndefinedMacro { name: (String, R) },
    ValueOutOfRange { value: i32, width: Width },
}

#[derive(Debug, PartialEq)]
pub enum Width {
    Byte,
}

use codebase::{BufId, BufRange};

pub struct DiagnosticsDumper;

impl DiagnosticsDumper {
    pub fn new() -> DiagnosticsDumper {
        DiagnosticsDumper {}
    }
}

impl DiagnosticsListener<(BufId, BufRange)> for DiagnosticsDumper {
    fn emit_diagnostic(&self, diagnostic: Diagnostic<(BufId, BufRange)>) {
        println!("{:?}", diagnostic)
    }
}

#[cfg(test)]
use codebase::{LineNumber, StringCodebase, TextBuf};

#[cfg(test)]
#[derive(Debug, PartialEq)]
struct Message<'a> {
    text: String,
    src_lines: Vec<(LineNumber, &'a str)>,
}

#[cfg(test)]
fn mk_diagnostic_message<'a>(
    diagnostic: Diagnostic<(BufId, BufRange)>,
    codebase: &'a StringCodebase,
) -> Message<'a> {
    let mut collectible_ranges = Vec::new();
    let text = match diagnostic {
        Diagnostic::UndefinedMacro { name } => {
            collectible_ranges.push(name.1);
            format!("invocation of undefined macro `{}`", name.0)
        }
        _ => panic!(),
    };
    let mut src_lines = Vec::new();
    for range_ref in collectible_ranges {
        let buf = codebase.buf(range_ref.0);
        let text_range = buf.text_range(&range_ref.1);
        src_lines.extend(buf.lines(text_range.start.line..text_range.end.line + 1))
    }
    Message { text, src_lines }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mk_message_for_undefined_macro() {
        let mut codebase = StringCodebase::new();
        let src = "    nop\n    my_macro a, $12\n\n";
        let buf_id = codebase.add_src_buf(src.to_string());
        let buf_range = BufRange::from(12..20);
        let diagnostic = Diagnostic::UndefinedMacro {
            name: ("my_macro".to_string(), (buf_id, buf_range)),
        };
        let message = mk_diagnostic_message(diagnostic, &codebase);
        assert_eq!(
            message,
            Message {
                text: "invocation of undefined macro `my_macro`".to_string(),
                src_lines: vec![(LineNumber(2), "    my_macro a, $12")],
            }
        )
    }
}
