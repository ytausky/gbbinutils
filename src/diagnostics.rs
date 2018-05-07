use codebase::{BufId, BufRange, LineNumber, TextBuf, TextCache};
use std::{io, cell::RefCell, rc::Rc};

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

pub trait DiagnosticsListener<TR> {
    fn emit_diagnostic(&self, diagnostic: Diagnostic<TR>);
}

#[derive(Debug, PartialEq)]
pub struct Diagnostic<TR> {
    pub message: Message,
    pub highlight: Option<TR>,
}

#[derive(Debug, PartialEq)]
pub enum Message {
    OperandCount { actual: usize, expected: usize },
    UndefinedMacro { name: String },
    ValueOutOfRange { value: i32, width: Width },
}

#[derive(Debug, PartialEq)]
pub enum Width {
    Byte,
}

pub struct TerminalDiagnostics<'a> {
    codebase: &'a RefCell<TextCache>,
}

impl<'a> TerminalDiagnostics<'a> {
    pub fn new(codebase: &'a RefCell<TextCache>) -> TerminalDiagnostics<'a> {
        TerminalDiagnostics { codebase }
    }
}

impl<'a> DiagnosticsListener<Rc<TokenRefData>> for TerminalDiagnostics<'a> {
    fn emit_diagnostic(&self, diagnostic: Diagnostic<Rc<TokenRefData>>) {
        let codebase = self.codebase.borrow();
        let elaborated_diagnostic = elaborate(diagnostic, &codebase);
        render(&elaborated_diagnostic, &mut io::stdout()).unwrap()
    }
}

#[derive(Debug, PartialEq)]
struct ElaboratedDiagnostic<'a> {
    text: String,
    src_lines: Vec<(LineNumber, &'a str)>,
}

fn elaborate<'a>(
    diagnostic: Diagnostic<Rc<TokenRefData>>,
    codebase: &'a TextCache,
) -> ElaboratedDiagnostic<'a> {
    let text = match diagnostic.message {
        Message::UndefinedMacro { name } => format!("invocation of undefined macro `{}`", name),
        _ => panic!(),
    };
    let mut src_lines = Vec::new();
    if let Some(rc_token_ref) = diagnostic.highlight {
        match *rc_token_ref {
            TokenRefData::Lexeme {
                ref range,
                ref context,
            } => {
                let buf = codebase.buf(context.buf_id);
                let text_range = buf.text_range(&range);
                src_lines.extend(buf.lines(text_range.start.line..text_range.end.line + 1))
            }
        }
    }
    ElaboratedDiagnostic { text, src_lines }
}

fn render<'a, W: io::Write>(
    elaborated_diagnostic: &ElaboratedDiagnostic<'a>,
    output: &mut W,
) -> io::Result<()> {
    output.write_all(elaborated_diagnostic.text.as_bytes())?;
    output.write_all("\n".as_bytes())?;
    for &(_, line) in elaborated_diagnostic.src_lines.iter() {
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
        let range = BufRange::from(12..20);
        let token_ref = Rc::new(TokenRefData::Lexeme {
            range,
            context: Rc::new(BufContextData {
                buf_id,
                included_from: None,
            }),
        });
        let diagnostic = Diagnostic {
            message: Message::UndefinedMacro {
                name: "my_macro".to_string(),
            },
            highlight: Some(token_ref),
        };
        let elaborated_diagnostic = elaborate(diagnostic, &codebase);
        assert_eq!(
            elaborated_diagnostic,
            ElaboratedDiagnostic {
                text: "invocation of undefined macro `my_macro`".to_string(),
                src_lines: vec![(LineNumber(2), "    my_macro a, $12")],
            }
        )
    }

    #[test]
    fn render_elaborated_diagnostic() {
        let elaborated_diagnostic = ElaboratedDiagnostic {
            text: "invocation of undefined macro `my_macro`".to_string(),
            src_lines: vec![(LineNumber(2), "    my_macro a, $12")],
        };
        let expected = "invocation of undefined macro `my_macro`\n    my_macro a, $12\n";
        let mut actual = Vec::new();
        render(&elaborated_diagnostic, &mut actual).unwrap();
        assert_eq!(String::from_utf8(actual).unwrap(), expected)
    }
}
