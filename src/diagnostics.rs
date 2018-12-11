use crate::backend::Width;
use crate::codebase::{BufId, CodebaseError, LineNumber, TextBuf, TextCache, TextRange};
use crate::instruction::IncDec;
use crate::span::{ContextFactory, MacroContextFactory, Merge, Span, SpanData};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::fmt;
#[cfg(test)]
use std::marker::PhantomData;

pub trait Diagnostics
where
    Self: DownstreamDiagnostics,
    Self: ContextFactory,
{
}

pub trait DownstreamDiagnostics
where
    Self: Span,
{
    type Output: Span<Span = Self::Span> + DiagnosticsListener + Merge;
    fn diagnostics(&mut self) -> &mut Self::Output;
}

impl<T> DownstreamDiagnostics for T
where
    T: DiagnosticsListener + Merge,
{
    type Output = Self;
    fn diagnostics(&mut self) -> &mut Self {
        self
    }
}

pub struct DiagnosticsSystem<C, O> {
    pub context: C,
    pub output: O,
}

impl<C, O> Span for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: DiagnosticsListener<Span = C::Span>,
{
    type Span = C::Span;
}

impl<C, O> Merge for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: DiagnosticsListener<Span = C::Span>,
{
    fn merge(&mut self, left: &Self::Span, right: &Self::Span) -> Self::Span {
        self.context.merge(left, right)
    }
}

impl<C, O> DiagnosticsListener for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: DiagnosticsListener<Span = C::Span>,
{
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<Self::Span>) {
        self.output.emit_diagnostic(diagnostic)
    }
}

impl<C, O> MacroContextFactory for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: DiagnosticsListener<Span = C::Span>,
{
    type MacroDefId = C::MacroDefId;
    type MacroExpansionContext = C::MacroExpansionContext;

    fn add_macro_def<P, B>(&mut self, name: Self::Span, params: P, body: B) -> Self::MacroDefId
    where
        P: IntoIterator<Item = Self::Span>,
        B: IntoIterator<Item = Self::Span>,
    {
        self.context.add_macro_def(name, params, body)
    }

    fn mk_macro_expansion_context<A, J>(
        &mut self,
        name: Self::Span,
        args: A,
        def: &Self::MacroDefId,
    ) -> Self::MacroExpansionContext
    where
        A: IntoIterator<Item = J>,
        J: IntoIterator<Item = Self::Span>,
    {
        self.context.mk_macro_expansion_context(name, args, def)
    }
}

impl<C, O> ContextFactory for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: DiagnosticsListener<Span = C::Span>,
{
    type BufContext = C::BufContext;

    fn mk_buf_context(
        &mut self,
        buf_id: BufId,
        included_from: Option<Self::Span>,
    ) -> Self::BufContext {
        self.context.mk_buf_context(buf_id, included_from)
    }
}

impl<C, O> Diagnostics for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: DiagnosticsListener<Span = C::Span>,
{
}

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
    Self: Span,
{
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<Self::Span>);
}

pub struct OutputForwarder<'a> {
    pub output: &'a mut dyn DiagnosticsOutput,
    pub codebase: &'a RefCell<TextCache>,
}

impl<'a> Span for OutputForwarder<'a> {
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
impl<S: Clone + fmt::Debug + PartialEq> Span for IgnoreDiagnostics<S> {
    type Span = S;
}

#[cfg(test)]
impl<S: Clone + fmt::Debug + PartialEq> DiagnosticsListener for IgnoreDiagnostics<S> {
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
impl Span for TestDiagnosticsListener {
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
    pub clauses: Vec<DiagnosticClause<T>>,
}

#[derive(Debug, PartialEq)]
pub struct DiagnosticClause<T> {
    pub file: T,
    pub tag: DiagnosticClauseTag,
    pub message: String,
    pub location: Option<DiagnosticLocation<T>>,
}

#[derive(Debug, PartialEq)]
pub enum DiagnosticClauseTag {
    Error,
}

impl fmt::Display for DiagnosticClauseTag {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            DiagnosticClauseTag::Error => "error",
        })
    }
}

#[derive(Debug, PartialEq)]
pub struct DiagnosticLocation<T> {
    pub line: LineNumber,
    pub source: T,
    pub highlight: Option<TextRange>,
}

pub fn mk_diagnostic(file: impl Into<String>, message: &Message<SpanData>) -> Diagnostic<String> {
    Diagnostic {
        clauses: vec![DiagnosticClause {
            file: file.into(),
            tag: DiagnosticClauseTag::Error,
            message: message.render(&TextCache::new()),
            location: None,
        }],
    }
}

impl InternalDiagnostic<SpanData> {
    fn elaborate<'a, T: From<&'a str>>(&self, codebase: &'a TextCache) -> Diagnostic<T> {
        match &self.highlight {
            SpanData::Buf { range, context } => {
                let buf = codebase.buf(context.buf_id);
                let highlight = buf.text_range(&range);
                let source = buf
                    .lines(highlight.start.line..=highlight.end.line)
                    .next()
                    .map(|(_, line)| line.trim_right())
                    .unwrap();
                Diagnostic {
                    clauses: vec![DiagnosticClause {
                        file: buf.name().into(),
                        tag: DiagnosticClauseTag::Error,
                        message: self.message.render(codebase),
                        location: Some(DiagnosticLocation {
                            line: highlight.start.line.into(),
                            source: source.into(),
                            highlight: Some(highlight),
                        }),
                    }],
                }
            }
            SpanData::Macro { .. } => unimplemented!(),
        }
    }
}

impl<T: Borrow<str>> fmt::Display for Diagnostic<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for clause in &self.clauses {
            clause.fmt(f)?
        }
        Ok(())
    }
}

impl<T: Borrow<str>> fmt::Display for DiagnosticClause<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.location {
            None => writeln!(f, "{}: {}: {}", self.file.borrow(), self.tag, self.message),
            Some(location) => {
                let squiggle = location
                    .highlight
                    .as_ref()
                    .map_or_else(String::new, mk_squiggle);
                writeln!(
                    f,
                    "{}:{}: {}: {}\n{}{}",
                    self.file.borrow(),
                    location.line,
                    self.tag,
                    self.message,
                    location.source.borrow(),
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

fn mk_snippet<'a>(codebase: &'a TextCache, span: &SpanData) -> &'a str {
    match span {
        SpanData::Buf { range, context } => &codebase.buf(context.buf_id).as_str()[range.clone()],
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
                clauses: vec![DiagnosticClause {
                    file: DUMMY_FILE,
                    tag: DiagnosticClauseTag::Error,
                    message: "invocation of undefined macro `my_macro`".to_string(),
                    location: Some(DiagnosticLocation {
                        line: LineNumber(2),
                        source: "    my_macro a, $12",
                        highlight: mk_highlight(LineNumber(2), 4, 12),
                    })
                }]
            }
        )
    }

    #[test]
    fn render_elaborated_diagnostic() {
        let elaborated_diagnostic = Diagnostic {
            clauses: vec![DiagnosticClause {
                file: DUMMY_FILE,
                tag: DiagnosticClauseTag::Error,
                message: "invocation of undefined macro `my_macro`".to_string(),
                location: Some(DiagnosticLocation {
                    line: LineNumber(2),
                    source: "    my_macro a, $12",
                    highlight: mk_highlight(LineNumber(2), 4, 12),
                }),
            }],
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
            clauses: vec![DiagnosticClause {
                file: DUMMY_FILE,
                tag: DiagnosticClauseTag::Error,
                message: "file constains invalid UTF-8".to_string(),
                location: None,
            }],
        };
        let expected = r"/my/file: error: file constains invalid UTF-8
";
        assert_eq!(diagnostic.to_string(), expected);
    }

    #[test]
    fn highlight_eof_with_one_tilde() {
        let elaborated = Diagnostic {
            clauses: vec![DiagnosticClause {
                file: DUMMY_FILE,
                tag: DiagnosticClauseTag::Error,
                message: "unexpected end of file".into(),
                location: Some(DiagnosticLocation {
                    line: LineNumber(2),
                    source: "dummy",
                    highlight: mk_highlight(LineNumber(2), 5, 5),
                }),
            }],
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
