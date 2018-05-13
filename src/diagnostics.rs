use codebase::{BufId, BufRange, LineNumber, TextBuf, TextCache, TextRange};
use std::{fmt, io, cell::RefCell, rc::Rc};

pub trait TokenTracker {
    type TokenRef: Clone + fmt::Debug + PartialEq;
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

#[derive(Debug, PartialEq)]
pub enum TokenRefData {
    Lexeme {
        range: BufRange,
        context: Rc<BufContextData>,
    },
}

#[derive(Debug, PartialEq)]
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
    message: Message,
    highlight: TR,
}

impl<TR> Diagnostic<TR> {
    pub fn new(message: Message, highlight: TR) -> Diagnostic<TR> {
        Diagnostic { message, highlight }
    }
}

#[derive(Debug, PartialEq)]
pub enum Message {
    OperandCount { actual: usize, expected: usize },
    UndefinedMacro { name: String },
    ValueOutOfRange { value: i32, width: Width },
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use diagnostics::Message::*;
        match self {
            OperandCount { actual, expected } => {
                write!(f, "expected {} operands, found {}", expected, actual)
            }
            UndefinedMacro { name } => write!(f, "invocation of undefined macro `{}`", name),
            ValueOutOfRange { value, width } => {
                write!(f, "value {} cannot be represented in a {}", value, width)
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Width {
    Byte,
}

impl fmt::Display for Width {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Width::Byte => f.write_str("byte"),
        }
    }
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
        let elaborated_diagnostic = elaborate(&diagnostic, &codebase);
        render(&elaborated_diagnostic, &mut io::stdout()).unwrap()
    }
}

#[derive(Debug, PartialEq)]
struct ElaboratedDiagnostic<'a> {
    text: String,
    buf_name: &'a str,
    highlight: TextRange,
    src_line: &'a str,
}

fn elaborate<'a>(
    diagnostic: &Diagnostic<Rc<TokenRefData>>,
    codebase: &'a TextCache,
) -> ElaboratedDiagnostic<'a> {
    match *diagnostic.highlight {
        TokenRefData::Lexeme {
            ref range,
            ref context,
        } => {
            let buf = codebase.buf(context.buf_id);
            let text_range = buf.text_range(&range);
            let (_, src_line) = buf.lines(text_range.start.line..text_range.end.line + 1)
                .next()
                .unwrap();
            ElaboratedDiagnostic {
                text: diagnostic.message.to_string(),
                buf_name: buf.name(),
                highlight: text_range,
                src_line,
            }
        }
    }
}

fn render<'a, W: io::Write>(
    diagnostic: &ElaboratedDiagnostic<'a>,
    output: &mut W,
) -> io::Result<()> {
    assert_eq!(
        diagnostic.highlight.start.line,
        diagnostic.highlight.end.line
    );
    let line_number: LineNumber = diagnostic.highlight.start.line.into();
    let mut highlight = String::new();
    let space_count = diagnostic.highlight.start.column_index;
    let tilde_count = diagnostic.highlight.end.column_index - space_count;
    for _ in 0..space_count {
        highlight.push(' ');
    }
    for _ in 0..tilde_count {
        highlight.push('~');
    }
    writeln!(
        output,
        "{}:{}: {}\n{}\n{}",
        diagnostic.buf_name, line_number, diagnostic.text, diagnostic.src_line, highlight
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use codebase::TextPosition;

    static DUMMY_FILE: &str = "/my/file";

    #[test]
    fn mk_message_for_undefined_macro() {
        let mut codebase = TextCache::new();
        let src = "    nop\n    my_macro a, $12\n\n";
        let buf_id = codebase.add_src_buf(DUMMY_FILE.into(), src.to_string());
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
            highlight: token_ref,
        };
        let elaborated_diagnostic = elaborate(&diagnostic, &codebase);
        assert_eq!(
            elaborated_diagnostic,
            ElaboratedDiagnostic {
                text: "invocation of undefined macro `my_macro`".to_string(),
                buf_name: DUMMY_FILE,
                highlight: mk_highlight(LineNumber(2), 4, 12),
                src_line: "    my_macro a, $12",
            }
        )
    }

    #[test]
    fn render_elaborated_diagnostic() {
        let elaborated_diagnostic = ElaboratedDiagnostic {
            text: "invocation of undefined macro `my_macro`".to_string(),
            buf_name: DUMMY_FILE,
            highlight: mk_highlight(LineNumber(2), 4, 12),
            src_line: "    my_macro a, $12",
        };
        let expected = r"/my/file:2: invocation of undefined macro `my_macro`
    my_macro a, $12
    ~~~~~~~~
";
        let mut actual = Vec::new();
        render(&elaborated_diagnostic, &mut actual).unwrap();
        assert_eq!(String::from_utf8(actual).unwrap(), expected)
    }

    fn mk_highlight(line_number: LineNumber, start: usize, end: usize) -> TextRange {
        TextRange {
            start: TextPosition {
                line: line_number.into(),
                column_index: start,
            },
            end: TextPosition {
                line: line_number.into(),
                column_index: end,
            },
        }
    }
}
