use codebase::{LineNumber, TextBuf, TextCache, TextRange};
use span::TokenRefData;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::fmt;
use Width;

pub trait DiagnosticsOutput {
    fn emit(&self, diagnostic: Diagnostic<String>);
}

pub struct TerminalOutput;

impl DiagnosticsOutput for TerminalOutput {
    fn emit(&self, diagnostic: Diagnostic<String>) {
        print!("{}", diagnostic)
    }
}

pub trait DiagnosticsListener<S> {
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<S>);
}

pub struct OutputForwarder<'a> {
    pub output: &'a mut dyn DiagnosticsOutput,
    pub codebase: &'a RefCell<TextCache>,
}

impl<'a> DiagnosticsListener<TokenRefData> for OutputForwarder<'a> {
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<TokenRefData>) {
        self.output
            .emit(diagnostic.elaborate(&self.codebase.borrow()))
    }
}

#[cfg(test)]
pub struct IgnoreDiagnostics;

#[cfg(test)]
impl<S> DiagnosticsListener<S> for IgnoreDiagnostics {
    fn emit_diagnostic(&mut self, _: InternalDiagnostic<S>) {}
}

#[cfg(test)]
pub struct TestDiagnosticsListener {
    pub diagnostics: RefCell<Vec<InternalDiagnostic<()>>>,
}

#[cfg(test)]
impl TestDiagnosticsListener {
    pub fn new() -> TestDiagnosticsListener {
        TestDiagnosticsListener {
            diagnostics: RefCell::new(Vec::new()),
        }
    }
}

#[cfg(test)]
impl DiagnosticsListener<()> for TestDiagnosticsListener {
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<()>) {
        self.diagnostics.borrow_mut().push(diagnostic)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InternalDiagnostic<S> {
    pub message: Message,
    pub spans: Vec<S>,
    pub highlight: S,
}

impl<S> InternalDiagnostic<S> {
    pub fn new(
        message: Message,
        spans: impl IntoIterator<Item = S>,
        highlight: S,
    ) -> InternalDiagnostic<S> {
        InternalDiagnostic {
            message,
            spans: spans.into_iter().collect(),
            highlight,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    AlwaysUnconditional,
    CannotDereference { category: KeywordOperandCategory },
    DestCannotBeConst,
    DestMustBeA,
    DestMustBeHl,
    IncompatibleOperand,
    KeywordInExpr,
    LdWidthMismatch { src: Width },
    MissingTarget,
    MustBeBit,
    MustBeConst,
    MustBeDeref,
    OperandCount { actual: usize, expected: usize },
    RequiresRegPair,
    RequiresSimpleOperand,
    SrcMustBeSp,
    StringInInstruction,
    UndefinedMacro { name: String },
    UnexpectedEof,
    UnexpectedToken,
    UnresolvedSymbol { symbol: String },
    ValueOutOfRange { value: i32, width: Width },
}

#[derive(Clone, Debug, PartialEq)]
pub enum KeywordOperandCategory {
    Reg,
    RegPair,
    ConditionCode,
}

impl Message {
    fn render<'a>(&self, snippets: impl IntoIterator<Item = &'a str>) -> String {
        use diagnostics::Message::*;
        let mut snippets = snippets.into_iter();
        let string = match self {
            AlwaysUnconditional => "instruction cannot be made conditional".into(),
            CannotDereference { category } => format!(
                "{} `{}` cannot be dereferenced",
                category,
                snippets.next().unwrap(),
            ),
            DestCannotBeConst => "destination operand cannot be a constant".into(),
            DestMustBeA => "destination of ALU operation must be `a`".into(),
            DestMustBeHl => "destination operand must be `hl`".into(),
            IncompatibleOperand => "operand cannot be used with this instruction".into(),
            KeywordInExpr => format!(
                "keyword `{}` cannot appear in expression",
                snippets.next().unwrap(),
            ),
            LdWidthMismatch { src } => {
                let (src_bits, dest_bits) = match src {
                    Width::Byte => (8, 16),
                    Width::Word => (16, 8),
                };
                format!(
                    "cannot load {}-bit source `{}` into {}-bit destination `{}`",
                    src_bits,
                    snippets.next().unwrap(),
                    dest_bits,
                    snippets.next().unwrap(),
                )
            }
            MissingTarget => "branch instruction requires target".into(),
            MustBeBit => format!(
                "first operand of `{}` must be bit number",
                snippets.next().unwrap()
            ),
            MustBeConst => "operand must be a constant".into(),
            MustBeDeref => format!(
                "operand `{}` must be dereferenced",
                snippets.next().unwrap()
            ),
            OperandCount { actual, expected } => format!(
                "expected {} operand{}, found {}",
                expected,
                pluralize(*expected),
                actual
            ),
            RequiresRegPair => "instruction requires a register pair".into(),
            RequiresSimpleOperand => "instruction requires 8-bit register or `(hl)`".into(),
            SrcMustBeSp => "source operand must be `sp`".into(),
            StringInInstruction => "strings cannot appear in instruction operands".into(),
            UndefinedMacro { name } => format!("invocation of undefined macro `{}`", name),
            UnexpectedEof => "unexpected end of file".into(),
            UnexpectedToken => format!(
                "encountered unexpected token `{}`",
                snippets.next().unwrap(),
            ),
            UnresolvedSymbol { symbol } => format!("symbol `{}` could not be resolved", symbol),
            ValueOutOfRange { value, width } => {
                format!("value {} cannot be represented in a {}", value, width)
            }
        };
        assert_eq!(snippets.next(), None);
        string
    }
}

impl fmt::Display for KeywordOperandCategory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            KeywordOperandCategory::Reg => f.write_str("register"),
            KeywordOperandCategory::RegPair => f.write_str("register pair"),
            KeywordOperandCategory::ConditionCode => f.write_str("condition code"),
        }
    }
}

fn pluralize(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

impl fmt::Display for Width {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Width::Byte => f.write_str("byte"),
            Width::Word => f.write_str("word"),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Diagnostic<T> {
    file: T,
    line: LineNumber,
    message: String,
    source: T,
    highlight: TextRange,
}

impl InternalDiagnostic<TokenRefData> {
    fn elaborate<'a, T: From<&'a str>>(&self, codebase: &'a TextCache) -> Diagnostic<T> {
        match self.highlight {
            TokenRefData::Lexeme {
                ref range,
                ref context,
            } => {
                let buf = codebase.buf(context.buf_id);
                let highlight = buf.text_range(&range);
                let (_, source) = buf
                    .lines(highlight.start.line..=highlight.end.line)
                    .next()
                    .unwrap();
                let snippets = self.spans.iter().map(|span| mk_snippet(codebase, span));
                Diagnostic {
                    file: buf.name().into(),
                    line: highlight.start.line.into(),
                    message: self.message.render(snippets),
                    source: source.into(),
                    highlight,
                }
            }
        }
    }
}

impl<T: Borrow<str>> fmt::Display for Diagnostic<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        assert_eq!(self.highlight.start.line, self.highlight.end.line);
        let mut highlight = String::new();
        let space_count = self.highlight.start.column_index;
        let tilde_count = match self.highlight.end.column_index - space_count {
            0 => 1,
            n => n,
        };
        for _ in 0..space_count {
            highlight.push(' ');
        }
        for _ in 0..tilde_count {
            highlight.push('~');
        }
        writeln!(
            f,
            "{}:{}: error: {}\n{}\n{}",
            self.file.borrow(),
            self.line,
            self.message,
            self.source.borrow(),
            highlight,
        )
    }
}

fn mk_snippet<'a>(codebase: &'a TextCache, span: &TokenRefData) -> &'a str {
    match span {
        TokenRefData::Lexeme { range, context } => {
            &codebase.buf(context.buf_id).as_str()[range.start..range.end]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codebase::{BufRange, TextPosition};
    use span::BufContextData;
    use std::rc::Rc;

    static DUMMY_FILE: &str = "/my/file";

    #[test]
    fn get_snippet() {
        let mut codebase = TextCache::new();
        let src = "add snippet, my";
        let buf_id = codebase.add_src_buf(DUMMY_FILE, src);
        let context = Rc::new(BufContextData {
            buf_id,
            included_from: None,
        });
        let span = TokenRefData::Lexeme {
            range: BufRange::from(4..11),
            context: context.clone(),
        };
        assert_eq!(mk_snippet(&codebase, &span), "snippet")
    }

    #[test]
    fn mk_message_for_undefined_macro() {
        let mut codebase = TextCache::new();
        let src = "    nop\n    my_macro a, $12\n\n";
        let buf_id = codebase.add_src_buf(DUMMY_FILE, src);
        let range = BufRange::from(12..20);
        let token_ref = TokenRefData::Lexeme {
            range,
            context: Rc::new(BufContextData {
                buf_id,
                included_from: None,
            }),
        };
        let diagnostic = InternalDiagnostic {
            message: Message::UndefinedMacro {
                name: "my_macro".to_string(),
            },
            spans: Vec::new(),
            highlight: token_ref,
        };
        assert_eq!(
            diagnostic.elaborate(&codebase),
            Diagnostic {
                file: DUMMY_FILE,
                line: LineNumber(2),
                message: "invocation of undefined macro `my_macro`".to_string(),
                source: "    my_macro a, $12",
                highlight: mk_highlight(LineNumber(2), 4, 12),
            }
        )
    }

    #[test]
    fn render_elaborated_diagnostic() {
        let elaborated_diagnostic = Diagnostic {
            file: DUMMY_FILE,
            line: LineNumber(2),
            message: "invocation of undefined macro `my_macro`".to_string(),
            source: "    my_macro a, $12",
            highlight: mk_highlight(LineNumber(2), 4, 12),
        };
        let expected = r"/my/file:2: error: invocation of undefined macro `my_macro`
    my_macro a, $12
    ~~~~~~~~
";
        assert_eq!(elaborated_diagnostic.to_string(), expected)
    }

    #[test]
    fn highlight_eof_with_one_tilde() {
        let elaborated = Diagnostic {
            file: DUMMY_FILE,
            line: LineNumber(2),
            message: "unexpected end of file".into(),
            source: "dummy",
            highlight: mk_highlight(LineNumber(2), 5, 5),
        };
        let expected = r"/my/file:2: error: unexpected end of file
dummy
     ~
";
        assert_eq!(elaborated.to_string(), expected)
    }

    #[test]
    fn expect_1_operand() {
        let message = Message::OperandCount {
            actual: 0,
            expected: 1,
        };
        assert_eq!(message.render(Vec::new()), "expected 1 operand, found 0")
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
