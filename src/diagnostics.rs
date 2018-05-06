use codebase::{BufId, BufRange};
use std::{fmt::Debug, rc::Rc};

#[cfg(test)]
use std::io::{self, Write};

pub trait TokenTracker {
    type TokenRef: Clone;
    type BufContext: Clone + LexemeRefFactory<TokenRef = Self::TokenRef>;
    fn mk_buf_context(
        &mut self,
        buf_id: BufId,
        included_from: Option<Self::TokenRef>,
    ) -> Self::BufContext;
}

pub trait LexemeRefFactory {
    type TokenRef;
    fn mk_lexeme_ref(&self, range: BufRange) -> Self::TokenRef;
}

#[derive(Debug)]
pub enum TokenRefData {
    Lexeme {
        range: BufRange,
        context: Rc<BufContextData>,
    },
}

#[derive(Debug)]
pub struct BufContextData {
    buf_id: BufId,
    included_from: Option<Rc<TokenRefData>>,
}

pub struct SimpleTokenTracker;

impl TokenTracker for SimpleTokenTracker {
    type TokenRef = Rc<TokenRefData>;
    type BufContext = SimpleBufTokenRefFactory;
    fn mk_buf_context(
        &mut self,
        buf_id: BufId,
        included_from: Option<Self::TokenRef>,
    ) -> Self::BufContext {
        let context = Rc::new(BufContextData {
            buf_id,
            included_from,
        });
        SimpleBufTokenRefFactory { context }
    }
}

#[derive(Clone)]
pub struct SimpleBufTokenRefFactory {
    context: Rc<BufContextData>,
}

impl LexemeRefFactory for SimpleBufTokenRefFactory {
    type TokenRef = Rc<TokenRefData>;
    fn mk_lexeme_ref(&self, range: BufRange) -> Self::TokenRef {
        Rc::new(TokenRefData::Lexeme {
            range,
            context: self.context.clone(),
        })
    }
}

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

pub struct DiagnosticsDumper;

impl DiagnosticsDumper {
    pub fn new() -> DiagnosticsDumper {
        DiagnosticsDumper {}
    }
}

impl<TR: Debug> DiagnosticsListener<TR> for DiagnosticsDumper {
    fn emit_diagnostic(&self, diagnostic: Diagnostic<TR>) {
        println!("{:?}", diagnostic)
    }
}

#[cfg(test)]
use codebase::{LineNumber, TextBuf, TextCache};

#[cfg(test)]
#[derive(Debug, PartialEq)]
struct Message<'a> {
    text: String,
    src_lines: Vec<(LineNumber, &'a str)>,
}

#[cfg(test)]
fn mk_diagnostic_message<'a>(
    diagnostic: Diagnostic<(BufId, BufRange)>,
    codebase: &'a TextCache,
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
fn render_message<'a, W: Write>(message: &Message<'a>, output: &mut W) -> io::Result<()> {
    output.write_all(message.text.as_bytes())?;
    output.write_all("\n".as_bytes())?;
    for &(_, line) in message.src_lines.iter() {
        output.write_all(line.as_bytes())?;
        output.write_all("\n".as_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mk_message_for_undefined_macro() {
        let mut codebase = TextCache::new();
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

    #[test]
    fn write_message() {
        let message = Message {
            text: "invocation of undefined macro `my_macro`".to_string(),
            src_lines: vec![(LineNumber(2), "    my_macro a, $12")],
        };
        let expected = "invocation of undefined macro `my_macro`\n    my_macro a, $12\n";
        let mut actual = Vec::new();
        render_message(&message, &mut actual).unwrap();
        assert_eq!(String::from_utf8(actual).unwrap(), expected)
    }
}
