use codebase::{BufId, BufRange, LineNumber, TextBuf, TextCache, TextRange};
use std::{cell::RefCell, cmp, fmt, rc::Rc};
use Width;

pub trait SourceInterval: Clone + fmt::Debug {
    fn extend(&self, other: &Self) -> Self;
}

#[cfg(test)]
impl SourceInterval for () {
    fn extend(&self, _: &Self) -> Self {}
}

pub trait Source {
    type Interval: SourceInterval;
    fn source_interval(&self) -> Self::Interval;
}

pub trait TokenTracker {
    type TokenRef: SourceInterval;
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

#[derive(Clone, Debug, PartialEq)]
pub enum TokenRefData {
    Lexeme {
        range: BufRange,
        context: Rc<BufContextData>,
    },
}

#[derive(Debug, PartialEq)]
pub struct BufContextData {
    buf_id: BufId,
    included_from: Option<TokenRefData>,
}

pub struct SimpleTokenTracker;

impl TokenTracker for SimpleTokenTracker {
    type TokenRef = TokenRefData;
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
    type TokenRef = TokenRefData;
    fn mk_lexeme_ref(&self, range: BufRange) -> Self::TokenRef {
        TokenRefData::Lexeme {
            range,
            context: self.context.clone(),
        }
    }
}

impl SourceInterval for TokenRefData {
    fn extend(&self, other: &Self) -> Self {
        use diagnostics::TokenRefData::*;
        match (self, other) {
            (
                Lexeme { range, context },
                Lexeme {
                    range: other_range,
                    context: other_context,
                },
            ) if Rc::ptr_eq(context, other_context) =>
            {
                Lexeme {
                    range: cmp::min(range.start, other_range.start)
                        ..cmp::max(range.end, other_range.end),
                    context: (*context).clone(),
                }
            }
            _ => panic!(),
        }
    }
}

pub trait DiagnosticsListener<TR> {
    fn emit_diagnostic(&self, diagnostic: Diagnostic<TR>);
}

#[derive(Debug, PartialEq)]
pub struct Diagnostic<SR> {
    message: Message<SR>,
    highlight: SR,
}

impl<SR> Diagnostic<SR> {
    pub fn new(message: Message<SR>, highlight: SR) -> Diagnostic<SR> {
        Diagnostic { message, highlight }
    }
}

#[derive(Debug, PartialEq)]
pub enum Message<SR> {
    AlwaysUnconditional,
    CannotDereference {
        category: KeywordOperandCategory,
        keyword: SR,
    },
    DestMustBeA,
    DestMustBeHl,
    IncompatibleOperand,
    KeywordInExpr {
        keyword: SR,
    },
    MissingTarget,
    OperandCount {
        actual: usize,
        expected: usize,
    },
    StringInInstruction,
    UndefinedMacro {
        name: String,
    },
    UnresolvedSymbol {
        symbol: String,
    },
    ValueOutOfRange {
        value: i32,
        width: Width,
    },
}

#[derive(Debug, PartialEq)]
pub enum KeywordOperandCategory {
    Reg,
    RegPair,
    ConditionCode,
}

impl Message<TokenRefData> {
    fn render(&self, codebase: &TextCache) -> String {
        use diagnostics::Message::*;
        match self {
            AlwaysUnconditional => "instruction cannot be made conditional".into(),
            CannotDereference { category, keyword } => format!(
                "{} `{}` cannot be dereferenced",
                category,
                mk_snippet(codebase, keyword)
            ),
            DestMustBeA => "destination of ALU operation must be `a`".into(),
            DestMustBeHl => "destination operand must be `hl`".into(),
            IncompatibleOperand => "operand cannot be used with this instruction".into(),
            KeywordInExpr { keyword } => format!(
                "keyword `{}` cannot appear in expression",
                mk_snippet(codebase, keyword)
            ),
            MissingTarget => "branch instruction requires target".into(),
            OperandCount { actual, expected } => format!(
                "expected {} operand{}, found {}",
                expected,
                pluralize(*expected),
                actual
            ),
            StringInInstruction => "strings cannot appear in instruction operands".into(),
            UndefinedMacro { name } => format!("invocation of undefined macro `{}`", name),
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

pub struct TerminalDiagnostics<'a> {
    codebase: &'a RefCell<TextCache>,
}

impl<'a> TerminalDiagnostics<'a> {
    pub fn new(codebase: &'a RefCell<TextCache>) -> TerminalDiagnostics<'a> {
        TerminalDiagnostics { codebase }
    }
}

impl<'a> DiagnosticsListener<TokenRefData> for TerminalDiagnostics<'a> {
    fn emit_diagnostic(&self, diagnostic: Diagnostic<TokenRefData>) {
        let codebase = self.codebase.borrow();
        let elaborated_diagnostic = elaborate(&diagnostic, &codebase);
        print!("{}", elaborated_diagnostic)
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
    diagnostic: &Diagnostic<TokenRefData>,
    codebase: &'a TextCache,
) -> ElaboratedDiagnostic<'a> {
    match diagnostic.highlight {
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
                text: diagnostic.message.render(codebase),
                buf_name: buf.name(),
                highlight: text_range,
                src_line,
            }
        }
    }
}

impl<'a> fmt::Display for ElaboratedDiagnostic<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        assert_eq!(self.highlight.start.line, self.highlight.end.line);
        let line_number: LineNumber = self.highlight.start.line.into();
        let mut highlight = String::new();
        let space_count = self.highlight.start.column_index;
        let tilde_count = self.highlight.end.column_index - space_count;
        for _ in 0..space_count {
            highlight.push(' ');
        }
        for _ in 0..tilde_count {
            highlight.push('~');
        }
        writeln!(
            f,
            "{}:{}: error: {}\n{}\n{}",
            self.buf_name, line_number, self.text, self.src_line, highlight
        )
    }
}

fn mk_snippet<'a>(codebase: &'a TextCache, interval: &TokenRefData) -> &'a str {
    match interval {
        TokenRefData::Lexeme { range, context } => {
            &codebase.buf(context.buf_id).as_str()[range.start..range.end]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codebase::TextPosition;

    static DUMMY_FILE: &str = "/my/file";

    #[test]
    fn extend_interval() {
        let mut codebase = TextCache::new();
        let src = "left right";
        let buf_id = codebase.add_src_buf(DUMMY_FILE, src);
        let context = Rc::new(BufContextData {
            buf_id,
            included_from: None,
        });
        let left = TokenRefData::Lexeme {
            range: BufRange::from(0..4),
            context: context.clone(),
        };
        let right = TokenRefData::Lexeme {
            range: BufRange::from(5..10),
            context: context.clone(),
        };
        let combined = left.extend(&right);
        assert_eq!(
            combined,
            TokenRefData::Lexeme {
                range: BufRange::from(0..10),
                context
            }
        )
    }

    #[test]
    fn get_snippet() {
        let mut codebase = TextCache::new();
        let src = "add snippet, my";
        let buf_id = codebase.add_src_buf(DUMMY_FILE, src);
        let context = Rc::new(BufContextData {
            buf_id,
            included_from: None,
        });
        let interval = TokenRefData::Lexeme {
            range: BufRange::from(4..11),
            context: context.clone(),
        };
        assert_eq!(mk_snippet(&codebase, &interval), "snippet")
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
        let expected = r"/my/file:2: error: invocation of undefined macro `my_macro`
    my_macro a, $12
    ~~~~~~~~
";
        assert_eq!(elaborated_diagnostic.to_string(), expected)
    }

    #[test]
    fn expect_1_operand() {
        let codebase = TextCache::new();
        let message = Message::OperandCount {
            actual: 0,
            expected: 1,
        };
        assert_eq!(message.render(&codebase), "expected 1 operand, found 0")
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
