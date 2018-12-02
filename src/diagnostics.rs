use crate::backend::Width;
use crate::codebase::{CodebaseError, LineNumber, TextBuf, TextCache, TextRange};
use crate::instruction::IncDec;
#[cfg(test)]
use crate::span::Span;
use crate::span::{HasSpan, SpanData};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::fmt;
#[cfg(test)]
use std::marker::PhantomData;

pub trait DiagnosticsOutput {
    fn emit(&mut self, diagnostic: Diagnostic<String>);
}

pub struct TerminalOutput;

impl DiagnosticsOutput for TerminalOutput {
    fn emit(&mut self, diagnostic: Diagnostic<String>) {
        print!("{}", diagnostic)
    }
}

pub trait DiagnosticsListener
where
    Self: HasSpan,
{
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<Self::Span>);
}

pub struct OutputForwarder<'a> {
    pub output: &'a mut dyn DiagnosticsOutput,
    pub codebase: &'a RefCell<TextCache>,
}

impl<'a> HasSpan for OutputForwarder<'a> {
    type Span = SpanData;
}

impl<'a> DiagnosticsListener for OutputForwarder<'a> {
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<SpanData>) {
        self.output
            .emit(diagnostic.elaborate(&self.codebase.borrow()))
    }
}

#[cfg(test)]
pub struct IgnoreDiagnostics<S>(PhantomData<S>);

#[cfg(test)]
impl<S> IgnoreDiagnostics<S> {
    pub fn new() -> Self {
        IgnoreDiagnostics(PhantomData)
    }
}

#[cfg(test)]
impl<S: Span> HasSpan for IgnoreDiagnostics<S> {
    type Span = S;
}

#[cfg(test)]
impl<S: Span> DiagnosticsListener for IgnoreDiagnostics<S> {
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
impl HasSpan for TestDiagnosticsListener {
    type Span = ();
}

#[cfg(test)]
impl DiagnosticsListener for TestDiagnosticsListener {
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<()>) {
        self.diagnostics.borrow_mut().push(diagnostic)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InternalDiagnostic<S> {
    pub message: Message<S>,
    pub highlight: S,
}

impl<S> InternalDiagnostic<S> {
    pub fn new(message: Message<S>, highlight: S) -> InternalDiagnostic<S> {
        InternalDiagnostic { message, highlight }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Message<S> {
    AfOutsideStackOperation,
    AlwaysUnconditional,
    CannotBeUsedAsTarget,
    CannotDereference {
        category: KeywordOperandCategory,
        operand: S,
    },
    CannotSpecifyTarget,
    ConditionOutsideBranch,
    DestCannotBeConst,
    DestMustBeA,
    DestMustBeHl,
    ExpectedString,
    IncompatibleOperand,
    InvalidUtf8,
    IoError {
        string: String,
    },
    KeywordInExpr {
        keyword: S,
    },
    LdDerefHlDerefHl {
        mnemonic: S,
        dest: S,
        src: S,
    },
    LdSpHlOperands,
    LdWidthMismatch {
        src_width: Width,
        src: S,
        dest: S,
    },
    MacroRequiresName,
    MissingTarget,
    MustBeBit {
        mnemonic: S,
    },
    MustBeConst,
    MustBeDeref {
        operand: S,
    },
    OnlySupportedByA,
    OperandCannotBeIncDec(IncDec),
    OperandCount {
        actual: usize,
        expected: usize,
    },
    RequiresConstantTarget {
        mnemonic: S,
    },
    RequiresRegPair,
    RequiresSimpleOperand,
    SrcMustBeSp,
    StringInInstruction,
    UndefinedMacro {
        name: String,
    },
    UnexpectedEof,
    UnexpectedToken {
        token: S,
    },
    UnmatchedParenthesis,
    UnresolvedSymbol {
        symbol: String,
    },
    ValueOutOfRange {
        value: i32,
        width: Width,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum KeywordOperandCategory {
    Reg,
    RegPair,
    ConditionCode,
}

impl<S> From<CodebaseError> for Message<S> {
    fn from(error: CodebaseError) -> Message<S> {
        match error {
            CodebaseError::IoError(error) => Message::IoError {
                string: error.to_string(),
            },
            CodebaseError::Utf8Error => Message::InvalidUtf8,
        }
    }
}

impl Message<SpanData> {
    fn render<'a>(&self, codebase: &'a TextCache) -> String {
        use crate::diagnostics::Message::*;
        match self {
            AfOutsideStackOperation => {
                "register pair `af` can only be used with `push` and `pop`".into()
            }
            AlwaysUnconditional => "instruction cannot be made conditional".into(),
            CannotBeUsedAsTarget => {
                "operand cannot be used as target for branching instructions".into()
            }
            CannotDereference { category, operand } => format!(
                "{} `{}` cannot be dereferenced",
                category,
                mk_snippet(codebase, operand),
            ),
            CannotSpecifyTarget => "branch target cannot be specified explicitly".into(),
            ConditionOutsideBranch => {
                "condition codes can only be used as operands for branching instructions".into()
            }
            DestCannotBeConst => "destination operand cannot be a constant".into(),
            DestMustBeA => "destination of ALU operation must be `a`".into(),
            DestMustBeHl => "destination operand must be `hl`".into(),
            ExpectedString => "expected string argument".into(),
            IncompatibleOperand => "operand cannot be used with this instruction".into(),
            InvalidUtf8 => "file contains invalid UTF-8".into(),
            IoError { string } => string.clone(),
            KeywordInExpr { keyword } => format!(
                "keyword `{}` cannot appear in expression",
                mk_snippet(codebase, keyword),
            ),
            LdDerefHlDerefHl {
                mnemonic,
                dest,
                src,
            } => format!(
                "`{} {}, {}` is not a legal instruction",
                mk_snippet(codebase, mnemonic),
                mk_snippet(codebase, dest),
                mk_snippet(codebase, src)
            ),
            LdSpHlOperands => {
                "the only legal 16-bit register to register transfer is from `hl` to `sp`".into()
            }
            LdWidthMismatch {
                src_width,
                src,
                dest,
            } => {
                let (src_bits, dest_bits) = match src_width {
                    Width::Byte => (8, 16),
                    Width::Word => (16, 8),
                };
                format!(
                    "cannot load {}-bit source `{}` into {}-bit destination `{}`",
                    src_bits,
                    mk_snippet(codebase, src),
                    dest_bits,
                    mk_snippet(codebase, dest),
                )
            }
            MacroRequiresName => "macro definition must be preceded by label".into(),
            MissingTarget => "branch instruction requires target".into(),
            MustBeBit { mnemonic } => format!(
                "first operand of `{}` must be bit number",
                mk_snippet(codebase, mnemonic),
            ),
            MustBeConst => "operand must be a constant".into(),
            MustBeDeref { operand } => format!(
                "operand `{}` must be dereferenced",
                mk_snippet(codebase, operand),
            ),
            OnlySupportedByA => "only `a` can be used for this operand".into(),
            OperandCannotBeIncDec(operation) => format!(
                "operand cannot be {}",
                match operation {
                    IncDec::Inc => "incremented",
                    IncDec::Dec => "decremented",
                }
            ),
            OperandCount { actual, expected } => format!(
                "expected {} operand{}, found {}",
                expected,
                pluralize(*expected),
                actual
            ),
            RequiresConstantTarget { mnemonic } => format!(
                "instruction `{}` requires a constant target",
                mk_snippet(codebase, mnemonic),
            ),
            RequiresRegPair => "instruction requires a register pair".into(),
            RequiresSimpleOperand => "instruction requires 8-bit register or `(hl)`".into(),
            SrcMustBeSp => "source operand must be `sp`".into(),
            StringInInstruction => "strings cannot appear in instruction operands".into(),
            UndefinedMacro { name } => format!("invocation of undefined macro `{}`", name),
            UnexpectedEof => "unexpected end of file".into(),
            UnexpectedToken { token } => format!(
                "encountered unexpected token `{}`",
                mk_snippet(codebase, token),
            ),
            UnmatchedParenthesis => "unmatched parenthesis".into(),
            UnresolvedSymbol { symbol } => format!("symbol `{}` could not be resolved", symbol),
            ValueOutOfRange { value, width } => {
                format!("value {} cannot be represented in a {}", value, width)
            }
        }
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
    pub file: T,
    pub message: String,
    pub location: Option<DiagnosticLocation<T>>,
}

#[derive(Debug, PartialEq)]
pub struct DiagnosticLocation<T> {
    pub line: LineNumber,
    pub source: T,
    pub highlight: TextRange,
}

pub fn mk_diagnostic(file: impl Into<String>, message: &Message<SpanData>) -> Diagnostic<String> {
    Diagnostic {
        file: file.into(),
        message: message.render(&TextCache::new()),
        location: None,
    }
}

impl InternalDiagnostic<SpanData> {
    fn elaborate<'a, T: From<&'a str>>(&self, codebase: &'a TextCache) -> Diagnostic<T> {
        match self.highlight {
            SpanData::Buf {
                ref range,
                ref context,
            } => {
                let buf = codebase.buf(context.buf_id);
                let highlight = buf.text_range(&range);
                let source = buf
                    .lines(highlight.start.line..=highlight.end.line)
                    .next()
                    .map(|(_, line)| line.trim_right())
                    .unwrap();
                Diagnostic {
                    file: buf.name().into(),
                    message: self.message.render(codebase),
                    location: Some(DiagnosticLocation {
                        line: highlight.start.line.into(),
                        source: source.into(),
                        highlight,
                    }),
                }
            }
            SpanData::Macro { .. } => unimplemented!(),
        }
    }
}

impl<T: Borrow<str>> fmt::Display for Diagnostic<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.location {
            None => writeln!(f, "{}: error: {}", self.file.borrow(), self.message),
            Some(location) => {
                assert_eq!(location.highlight.start.line, location.highlight.end.line);
                let mut highlight = String::new();
                let space_count = location.highlight.start.column_index;
                let tilde_count = match location.highlight.end.column_index - space_count {
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
                    location.line,
                    self.message,
                    location.source.borrow(),
                    highlight,
                )
            }
        }
    }
}

fn mk_snippet<'a>(codebase: &'a TextCache, span: &SpanData) -> &'a str {
    match span {
        SpanData::Buf { range, context } => {
            &codebase.buf(context.buf_id).as_str()[range.start..range.end]
        }
        SpanData::Macro { .. } => unimplemented!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebase::{BufRange, TextPosition};
    use crate::span::BufContextData;
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
        let span = SpanData::Buf {
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
        let token_ref = SpanData::Buf {
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
            highlight: token_ref,
        };
        assert_eq!(
            diagnostic.elaborate(&codebase),
            Diagnostic {
                file: DUMMY_FILE,
                message: "invocation of undefined macro `my_macro`".to_string(),
                location: Some(DiagnosticLocation {
                    line: LineNumber(2),
                    source: "    my_macro a, $12",
                    highlight: mk_highlight(LineNumber(2), 4, 12),
                })
            }
        )
    }

    #[test]
    fn render_elaborated_diagnostic() {
        let elaborated_diagnostic = Diagnostic {
            file: DUMMY_FILE,
            message: "invocation of undefined macro `my_macro`".to_string(),
            location: Some(DiagnosticLocation {
                line: LineNumber(2),
                source: "    my_macro a, $12",
                highlight: mk_highlight(LineNumber(2), 4, 12),
            }),
        };
        let expected = r"/my/file:2: error: invocation of undefined macro `my_macro`
    my_macro a, $12
    ~~~~~~~~
";
        assert_eq!(elaborated_diagnostic.to_string(), expected)
    }

    #[test]
    fn render_diagnostic_without_source() {
        let diagnostic = Diagnostic {
            file: DUMMY_FILE,
            message: "file constains invalid UTF-8".to_string(),
            location: None,
        };
        let expected = r"/my/file: error: file constains invalid UTF-8
";
        assert_eq!(diagnostic.to_string(), expected);
    }

    #[test]
    fn highlight_eof_with_one_tilde() {
        let elaborated = Diagnostic {
            file: DUMMY_FILE,
            message: "unexpected end of file".into(),
            location: Some(DiagnosticLocation {
                line: LineNumber(2),
                source: "dummy",
                highlight: mk_highlight(LineNumber(2), 5, 5),
            }),
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
        assert_eq!(
            message.render(&TextCache::new()),
            "expected 1 operand, found 0"
        )
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
