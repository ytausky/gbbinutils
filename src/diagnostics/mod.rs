pub use self::message::{KeywordOperandCategory, Message};
use crate::codebase::{BufId, LineNumber, TextBuf, TextCache, TextRange};
use crate::span::*;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::fmt;
#[cfg(test)]
use std::marker::PhantomData;

mod message;
pub mod span;

pub trait Diagnostics: DownstreamDiagnostics + ContextFactory {}

pub trait DownstreamDiagnostics: EmitDiagnostic + MergeSpans + MkSnippetRef {}

impl<T: EmitDiagnostic + MergeSpans + MkSnippetRef> DownstreamDiagnostics for T {}

pub trait DelegateDiagnostics {
    type Delegate: DownstreamDiagnostics;

    fn diagnostics(&mut self) -> &mut Self::Delegate;
}

impl<T: DelegateDiagnostics> Span for T {
    type Span = <T::Delegate as Span>::Span;
}

impl<T: DelegateDiagnostics> SnippetRef for T {
    type SnippetRef = <T::Delegate as SnippetRef>::SnippetRef;
}

impl<T: DelegateDiagnostics> MergeSpans for T {
    fn merge_spans(&mut self, left: &Self::Span, right: &Self::Span) -> Self::Span {
        self.diagnostics().merge_spans(left, right)
    }
}

impl<T: DelegateDiagnostics> MkSnippetRef for T {
    fn mk_snippet_ref(&mut self, span: &Self::Span) -> Self::SnippetRef {
        self.diagnostics().mk_snippet_ref(span)
    }
}

impl<T: DelegateDiagnostics> EmitDiagnostic for T {
    fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<Self::Span, Self::SnippetRef>) {
        self.diagnostics().emit_diagnostic(diagnostic)
    }
}

pub struct DiagnosticsSystem<C, O> {
    pub context: C,
    pub output: O,
}

impl<C, O> Span for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: EmitDiagnostic<Span = C::Span>,
{
    type Span = C::Span;
}

impl<C, O> SnippetRef for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: EmitDiagnostic<Span = C::Span, SnippetRef = C::SnippetRef>,
{
    type SnippetRef = C::SnippetRef;
}

impl<C, O> MergeSpans for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: EmitDiagnostic<Span = C::Span>,
{
    fn merge_spans(&mut self, left: &Self::Span, right: &Self::Span) -> Self::Span {
        self.context.merge_spans(left, right)
    }
}

impl<C, O> MkSnippetRef for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: EmitDiagnostic<Span = C::Span, SnippetRef = C::SnippetRef>,
{
    fn mk_snippet_ref(&mut self, span: &Self::Span) -> Self::SnippetRef {
        self.context.mk_snippet_ref(span)
    }
}

impl<C, O> EmitDiagnostic for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: EmitDiagnostic<Span = C::Span, SnippetRef = C::SnippetRef>,
{
    fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<Self::Span, Self::SnippetRef>) {
        self.output.emit_diagnostic(diagnostic)
    }
}

impl<C, O> MacroContextFactory for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: EmitDiagnostic<Span = C::Span, SnippetRef = C::SnippetRef>,
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
    O: EmitDiagnostic<Span = C::Span, SnippetRef = C::SnippetRef>,
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
    O: EmitDiagnostic<Span = C::Span, SnippetRef = C::SnippetRef>,
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

pub trait EmitDiagnostic: SnippetRef + Span {
    fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<Self::Span, Self::SnippetRef>);
}

pub struct OutputForwarder<'a> {
    pub output: &'a mut dyn DiagnosticsOutput,
    pub codebase: &'a RefCell<TextCache>,
}

impl<'a> SnippetRef for OutputForwarder<'a> {
    type SnippetRef = BufSnippetRef;
}

impl<'a> Span for OutputForwarder<'a> {
    type Span = SpanData;
}

impl<'a> EmitDiagnostic for OutputForwarder<'a> {
    fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<SpanData, BufSnippetRef>) {
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
impl<S> SnippetRef for IgnoreDiagnostics<S> {
    type SnippetRef = S;
}

#[cfg(test)]
impl<S: Clone + fmt::Debug + PartialEq> EmitDiagnostic for IgnoreDiagnostics<S> {
    fn emit_diagnostic(&mut self, _: CompactDiagnostic<S, S>) {}
}

#[cfg(test)]
pub struct TestDiagnosticsListener {
    pub diagnostics: RefCell<Vec<CompactDiagnostic<(), ()>>>,
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
impl SnippetRef for TestDiagnosticsListener {
    type SnippetRef = ();
}

#[cfg(test)]
impl EmitDiagnostic for TestDiagnosticsListener {
    fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<(), ()>) {
        self.diagnostics.borrow_mut().push(diagnostic)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompactDiagnostic<S, R> {
    pub message: Message<R>,
    pub highlight: S,
}

impl<S, R> CompactDiagnostic<S, R> {
    pub fn new(message: Message<R>, highlight: S) -> CompactDiagnostic<S, R> {
        CompactDiagnostic { message, highlight }
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

pub fn mk_diagnostic(
    file: impl Into<String>,
    message: &Message<BufSnippetRef>,
) -> Diagnostic<String> {
    Diagnostic {
        clauses: vec![DiagnosticClause {
            file: file.into(),
            tag: DiagnosticClauseTag::Error,
            message: message.render(&TextCache::new()),
            location: None,
        }],
    }
}

impl CompactDiagnostic<SpanData, BufSnippetRef> {
    fn elaborate<'a, T: From<&'a str>>(&self, codebase: &'a TextCache) -> Diagnostic<T> {
        Diagnostic {
            clauses: vec![self.main_clause(codebase)],
        }
    }

    fn main_clause<'a, T: From<&'a str>>(&self, codebase: &'a TextCache) -> DiagnosticClause<T> {
        let (file, source, highlight) = match &self.highlight {
            SpanData::Buf { range, context } => {
                let buf = codebase.buf(context.buf_id);
                let highlight = buf.text_range(&range);
                let source = buf
                    .lines(highlight.start.line..=highlight.end.line)
                    .next()
                    .map(|(_, line)| line.trim_right())
                    .unwrap();
                (buf.name().into(), source, highlight)
            }
            SpanData::Macro { .. } => unimplemented!(),
        };
        DiagnosticClause {
            file,
            tag: DiagnosticClauseTag::Error,
            message: self.message.render(codebase),
            location: Some(DiagnosticLocation {
                line: highlight.start.line.into(),
                source: source.into(),
                highlight: Some(highlight),
            }),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebase::{BufRange, TextPosition};
    use crate::span::BufContextData;
    use std::rc::Rc;

    static DUMMY_FILE: &str = "/my/file";

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
        let diagnostic = CompactDiagnostic {
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
