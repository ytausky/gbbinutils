//! Types comprising the assembler's diagnostic output.
//!
//! During assembly the assembler may emit any number of diagnostic messages, each of which is
//! encoded in a [`Diagnostic`](struct.Diagnostic.html) along with all the information necessary to
//! present it to the end user.

pub(crate) use self::message::{KeywordOperandCategory, Message, ValueKind};

use crate::codebase::{Codebase, TextBuf, TextCache};
#[cfg(test)]
use crate::log::Log;
use crate::object::{SourceFileId, SourceFileRange, Span, SpanData};
#[cfg(test)]
use crate::span::fake::FakeSpanSystem;
use crate::span::*;

use std::fmt;
use std::ops::{Add, AddAssign};

mod message;

#[cfg(test)]
pub(crate) use self::mock::*;

/// A full description of an assembler diagnostic.
///
/// A [`Diagnostic`](struct.Diagnostic.html) contains all the information required to display a
/// meaningful diagnostic message to a user. It consists of one or more
/// [`Clause`](struct.Clause.html)s, where the first one describes the object of the diagnostic and
/// the rest provide more context if necessary.
#[derive(Debug, PartialEq)]
pub struct Diagnostic {
    pub clauses: Vec<Clause>,
}

/// A single clause of an assembler diagnostic.
///
/// A [`Clause`](struct.Clause.html) contains a message that's relevant to a particular source file
/// or a part thereof. In addition it may include an excerpt from said file, as well as an optional
/// highlight within it.
#[derive(Debug, PartialEq)]
pub struct Clause {
    pub file: String,
    pub tag: Tag,
    pub message: String,
    pub excerpt: Option<Excerpt>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tag {
    Error,
    Note,
}

/// A code excerpt with an optional highlight.
///
/// An [`Excerpt`](struct.Excerpt.html) contains a single line of code meant to provide context for
/// a diagnostic clause. The optional highlight can pinpoint the interesting part of the line.
#[derive(Debug, PartialEq)]
pub struct Excerpt {
    pub line: LineNumber,
    pub source: String,
    pub highlight: Option<TextRange>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineNumber(pub usize);

#[derive(Debug, PartialEq)]
pub struct TextPosition {
    pub line: LineIndex,
    pub column_index: usize,
}

#[derive(Debug, PartialEq)]
pub struct TextRange {
    pub start: TextPosition,
    pub end: TextPosition,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct LineIndex(pub usize);

impl Add<usize> for LineIndex {
    type Output = LineIndex;
    fn add(mut self, rhs: usize) -> Self::Output {
        self += rhs;
        self
    }
}

impl AddAssign<usize> for LineIndex {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs
    }
}

impl From<LineIndex> for LineNumber {
    fn from(LineIndex(index): LineIndex) -> LineNumber {
        LineNumber(index + 1)
    }
}

impl From<LineNumber> for LineIndex {
    fn from(LineNumber(n): LineNumber) -> LineIndex {
        assert_ne!(n, 0);
        LineIndex(n - 1)
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for clause in &self.clauses {
            write!(f, "{}", clause)?
        }
        Ok(())
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.excerpt {
            None => writeln!(f, "{}: {}: {}", self.file, self.tag, self.message),
            Some(location) => {
                let squiggle = location
                    .highlight
                    .as_ref()
                    .map_or_else(String::new, mk_squiggle);
                writeln!(
                    f,
                    "{}:{}: {}: {}\n{}{}",
                    self.file, location.line, self.tag, self.message, location.source, squiggle,
                )
            }
        }
    }
}

impl fmt::Display for LineNumber {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.0.fmt(formatter)
    }
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Tag::Error => "error",
            Tag::Note => "note",
        })
    }
}

fn mk_squiggle(range: &TextRange) -> String {
    assert_eq!(range.start.line, range.end.line);

    use std::cmp::max;
    let space_count = range.start.column_index;
    let caret_count = max(range.end.column_index - space_count, 1);

    use std::iter::{once, repeat};
    let spaces = repeat(' ').take(space_count);
    let carets = repeat('^').take(caret_count);
    once('\n').chain(spaces).chain(carets).collect()
}

pub(crate) trait Diagnostics<S>
where
    Self: MergeSpans<S> + BackendDiagnostics<S>,
{
}

pub(crate) trait BackendDiagnostics<S>
where
    Self: StripSpan<S> + EmitDiag<S, <Self as StripSpan<S>>::Stripped>,
{
}

impl<T, S> BackendDiagnostics<S> for T where
    T: StripSpan<S> + EmitDiag<S, <T as StripSpan<S>>::Stripped>
{
}

impl<T, S> Diagnostics<S> for T where T: MergeSpans<S> + BackendDiagnostics<S> {}

pub trait DiagnosticsOutput {
    fn emit(&mut self, diagnostic: Diagnostic);
}

pub(crate) trait EmitDiag<S, T> {
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<S, T>>);
}

pub(crate) struct OutputForwarder<'a> {
    pub output: &'a mut dyn FnMut(Diagnostic),
}

pub(crate) struct DiagnosticsContext<'a, C, R, D> {
    pub codebase: &'a mut C,
    pub registry: &'a mut R,
    pub diagnostics: &'a mut D,
}

impl<'a, C, R, D, S> MergeSpans<S> for DiagnosticsContext<'a, C, R, D>
where
    R: MergeSpans<S>,
{
    fn merge_spans(&mut self, left: &S, right: &S) -> S {
        self.registry.merge_spans(left, right)
    }
}

impl<'a, C, R, D, S> StripSpan<S> for DiagnosticsContext<'a, C, R, D>
where
    R: StripSpan<S>,
{
    type Stripped = R::Stripped;

    fn strip_span(&mut self, span: &S) -> Self::Stripped {
        self.registry.strip_span(span)
    }
}

impl<'a, 'b> EmitDiag<Span, StrippedBufSpan>
    for DiagnosticsContext<'b, Codebase<'a>, SpanData, OutputForwarder<'a>>
{
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<Span, StrippedBufSpan>>) {
        (self.diagnostics.output)(
            diag.into()
                .expand(self.registry)
                .render(&self.codebase.cache.borrow()),
        )
    }
}

#[cfg(test)]
impl<'a, 'b, S> EmitDiag<S, S>
    for DiagnosticsContext<'b, Codebase<'a>, FakeSpanSystem<S>, OutputForwarder<'a>>
{
    fn emit_diag(&mut self, _: impl Into<CompactDiag<S, S>>) {}
}

pub(crate) struct IgnoreDiagnostics;

impl<'a, C, R, S, T> EmitDiag<S, T> for DiagnosticsContext<'a, C, R, IgnoreDiagnostics> {
    fn emit_diag(&mut self, _: impl Into<CompactDiag<S, T>>) {}
}

impl<S: Clone> StripSpan<S> for IgnoreDiagnostics {
    type Stripped = S;

    fn strip_span(&mut self, span: &S) -> Self::Stripped {
        span.clone()
    }
}

impl<S: Clone> EmitDiag<S, S> for IgnoreDiagnostics {
    fn emit_diag(&mut self, _: impl Into<CompactDiag<S>>) {}
}

#[cfg(test)]
pub(crate) struct TestDiagnosticsListener<S> {
    pub diagnostics: Log<CompactDiag<S>>,
}

#[cfg(test)]
impl<S> TestDiagnosticsListener<S> {
    pub fn new() -> TestDiagnosticsListener<S> {
        TestDiagnosticsListener {
            diagnostics: Log::default(),
        }
    }
}

#[cfg(test)]
impl<S: Clone> SpanSource for TestDiagnosticsListener<S> {
    type Span = S;
}

#[cfg(test)]
impl<S: Clone> MergeSpans<S> for TestDiagnosticsListener<S> {
    fn merge_spans(&mut self, _left: &S, _right: &S) -> S {
        unimplemented!()
    }
}

#[cfg(test)]
impl<S: Clone> StripSpan<S> for TestDiagnosticsListener<S> {
    type Stripped = S;

    fn strip_span(&mut self, span: &S) -> Self::Stripped {
        span.clone()
    }
}

#[cfg(test)]
impl<'a, C, R, S> EmitDiag<S, S> for DiagnosticsContext<'a, C, R, TestDiagnosticsListener<S>> {
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<S>>) {
        self.diagnostics.diagnostics.push(diag.into())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompactDiag<S, R = S> {
    main: CompactClause<S, R>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompactClause<S, R> {
    pub message: Message<R>,
    pub highlight: S,
}

impl<S, R> From<CompactClause<S, R>> for CompactDiag<S, R> {
    fn from(clause: CompactClause<S, R>) -> Self {
        CompactDiag { main: clause }
    }
}

impl<R> Message<R> {
    pub(crate) fn at<S>(self, highlight: S) -> CompactClause<S, R> {
        CompactClause {
            message: self,
            highlight,
        }
    }
}

#[derive(Debug, PartialEq)]
struct ExpandedDiagnostic {
    clauses: Vec<ExpandedDiagnosticClause>,
}

#[derive(Debug, PartialEq)]
struct ExpandedDiagnosticClause {
    buf_id: SourceFileId,
    tag: Tag,
    message: Message<StrippedBufSpan>,
    location: Option<SourceFileRange>,
}

impl CompactDiag<Span, StrippedBufSpan> {
    fn expand(self, registry: &mut SpanData) -> ExpandedDiagnostic {
        let StrippedBufSpan { buf_id, range } = registry.strip_span(&self.main.highlight);
        let main_clause = ExpandedDiagnosticClause {
            buf_id,
            tag: Tag::Error,
            message: self.main.message,
            location: Some(range),
        };
        let mut clauses = vec![main_clause];
        if let Some(note) = mk_called_here_clause(&self.main.highlight, registry) {
            clauses.push(note)
        }
        ExpandedDiagnostic { clauses }
    }
}

fn mk_called_here_clause(span: &Span, registry: &mut SpanData) -> Option<ExpandedDiagnosticClause> {
    let call = if let Span::MacroExpansion { metadata, .. } = span {
        registry.macro_expansions[metadata.0].name_span.clone()
    } else {
        return None;
    };
    let stripped = registry.strip_span(&call);
    Some(ExpandedDiagnosticClause {
        buf_id: stripped.buf_id,
        tag: Tag::Note,
        location: Some(stripped.range.clone()),
        message: Message::CalledHere { name: stripped },
    })
}

impl ExpandedDiagnostic {
    fn render(&self, codebase: &TextCache) -> Diagnostic {
        Diagnostic {
            clauses: self
                .clauses
                .iter()
                .map(|clause| clause.render(codebase))
                .collect(),
        }
    }
}

impl ExpandedDiagnosticClause {
    fn render(&self, codebase: &TextCache) -> Clause {
        let buf = codebase.buf(self.buf_id);
        let excerpt = self.location.as_ref().map(|range| {
            let highlight = buf.text_range(&range);
            let source = buf
                .lines(highlight.start.line..=highlight.end.line)
                .next()
                .map(|(_, line)| line.trim_end())
                .unwrap()
                .into();
            Excerpt {
                line: highlight.start.line.into(),
                source,
                highlight: Some(highlight),
            }
        });
        Clause {
            file: buf.name().into(),
            tag: self.tag,
            message: self.message.render(codebase),
            excerpt,
        }
    }
}

#[cfg(test)]
pub(crate) mod mock {
    use super::*;

    use crate::log::Log;

    use std::fmt::Debug;
    use std::marker::PhantomData;

    pub(crate) trait Merge: Sized {
        fn merge(left: impl Into<Self>, right: impl Into<Self>) -> Self;
    }

    #[derive(Clone, Debug, PartialEq)]
    pub(crate) enum MockSpan<T> {
        Basic(T),
        Merge(Box<Self>, Box<Self>),
    }

    impl<T> Default for MockSpan<T> {
        fn default() -> Self {
            panic!()
        }
    }

    impl<T> From<T> for MockSpan<T> {
        fn from(token: T) -> Self {
            MockSpan::Basic(token)
        }
    }

    impl Merge for () {
        fn merge(_: impl Into<Self>, _: impl Into<Self>) -> Self {}
    }

    impl<T> Merge for MockSpan<T> {
        fn merge(left: impl Into<Self>, right: impl Into<Self>) -> Self {
            MockSpan::Merge(Box::new(left.into()), Box::new(right.into()))
        }
    }

    pub(crate) struct MockDiagnostics<T, S> {
        log: Log<T>,
        _span: PhantomData<S>,
    }

    impl<T, S> MockDiagnostics<T, S> {
        pub fn new(log: Log<T>) -> Self {
            Self {
                log,
                _span: PhantomData,
            }
        }
    }

    impl<T, S> Clone for MockDiagnostics<T, S> {
        fn clone(&self) -> Self {
            Self::new(self.log.clone())
        }
    }

    #[derive(Debug, PartialEq)]
    pub enum DiagnosticsEvent<S> {
        EmitDiag(CompactDiag<S>),
    }

    impl<S> From<CompactDiag<S>> for DiagnosticsEvent<S> {
        fn from(diag: CompactDiag<S>) -> Self {
            DiagnosticsEvent::EmitDiag(diag)
        }
    }

    impl<T, S: Clone> SpanSource for MockDiagnostics<T, S> {
        type Span = S;
    }

    impl<T, S: Clone> StripSpan<S> for MockDiagnostics<T, S> {
        type Stripped = S;

        fn strip_span(&mut self, span: &S) -> Self::Stripped {
            span.clone()
        }
    }

    impl<T, S: Clone + Merge> MergeSpans<S> for MockDiagnostics<T, S> {
        fn merge_spans(&mut self, left: &S, right: &S) -> S {
            S::merge(left.clone(), right.clone())
        }
    }

    impl<'a, C, R, T, S> EmitDiag<S, S> for DiagnosticsContext<'a, C, R, MockDiagnostics<T, S>>
    where
        T: From<DiagnosticsEvent<S>>,
    {
        fn emit_diag(&mut self, diag: impl Into<CompactDiag<S>>) {
            self.diagnostics
                .log
                .push(DiagnosticsEvent::EmitDiag(diag.into()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::object::*;

    static DUMMY_FILE: &str = "/my/file";

    #[test]
    fn mk_message_for_not_a_mnemonic() {
        let mut codebase = TextCache::new();
        let src = "    nop\n    my_macro a, $12\n\n";
        let buf_id = codebase.add_src_buf(DUMMY_FILE, src);
        let mut spans = SpanData::default();
        let inclusion_metadata = spans.add_file_inclusion(FileInclusionMetadata {
            file: buf_id,
            from: None,
        });
        let range = 12..20;
        let token_span = spans.encode_span(Span::SourceFile {
            inclusion_metadata,
            range: range.clone(),
        });
        let diagnostic = CompactDiag::from(
            Message::NotAMnemonic {
                name: StrippedBufSpan { buf_id, range },
            }
            .at(token_span),
        );
        assert_eq!(
            diagnostic.expand(&mut spans).render(&codebase),
            Diagnostic {
                clauses: vec![Clause {
                    file: DUMMY_FILE.to_string(),
                    tag: Tag::Error,
                    message: "`my_macro` is not a mnemonic".to_string(),
                    excerpt: Some(Excerpt {
                        line: LineNumber(2),
                        source: "    my_macro a, $12".to_string(),
                        highlight: mk_highlight(LineNumber(2), 4, 12),
                    })
                }]
            }
        )
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

    #[test]
    fn expand_error_in_macro() {
        let buf_id = SourceFileId(0);
        let mut spans = SpanData::default();
        let inclusion_metadata = spans.add_file_inclusion(FileInclusionMetadata {
            file: buf_id,
            from: None,
        });
        let def_name_span = spans.encode_span(Span::SourceFile {
            inclusion_metadata,
            range: 0..1,
        });
        let body_span = spans.encode_span(Span::SourceFile {
            inclusion_metadata,
            range: 2..3,
        });
        let macro_def = spans.add_macro_def(MacroDefMetadata {
            name_span: def_name_span,
            param_spans: Box::new([]),
            body_spans: Box::new([body_span]),
        });
        let call_range = 10..11;
        let expansion_name_span = spans.encode_span(Span::SourceFile {
            inclusion_metadata,
            range: call_range.clone(),
        });
        let metadata = spans.add_macro_expansion(MacroExpansionMetadata {
            name_span: expansion_name_span,
            def: macro_def,
            arg_spans: Box::new([]),
        });
        let position = MacroExpansionPos {
            token: 0,
            param_expansion: None,
        };
        let span = spans.encode_span(Span::MacroExpansion {
            metadata,
            range: position.clone()..=position,
        });
        let message = Message::AfOutsideStackOperation;
        let compact = CompactDiag::from(message.clone().at(span));
        let expected = ExpandedDiagnostic {
            clauses: vec![
                ExpandedDiagnosticClause {
                    buf_id,
                    tag: Tag::Error,
                    message,
                    location: Some(2..3),
                },
                ExpandedDiagnosticClause {
                    buf_id,
                    tag: Tag::Note,
                    message: Message::CalledHere {
                        name: StrippedBufSpan {
                            buf_id,
                            range: call_range,
                        },
                    },
                    location: Some(10..11),
                },
            ],
        };
        assert_eq!(compact.expand(&mut spans), expected)
    }

    #[test]
    fn render_elaborated_diagnostic() {
        let diagnostic = Diagnostic {
            clauses: vec![Clause {
                file: DUMMY_FILE.to_string(),
                tag: Tag::Error,
                message: "invocation of undefined macro `my_macro`".to_string(),
                excerpt: Some(Excerpt {
                    line: LineNumber(2),
                    source: "    my_macro a, $12".to_string(),
                    highlight: mk_highlight(LineNumber(2), 4, 12),
                }),
            }],
        };
        let expected = r"/my/file:2: error: invocation of undefined macro `my_macro`
    my_macro a, $12
    ^^^^^^^^
";
        assert_eq!(diagnostic.to_string(), expected)
    }

    #[test]
    fn render_diagnostic_without_source() {
        let diagnostic = Diagnostic {
            clauses: vec![Clause {
                file: DUMMY_FILE.to_string(),
                tag: Tag::Error,
                message: "file constains invalid UTF-8".to_string(),
                excerpt: None,
            }],
        };
        let expected = r"/my/file: error: file constains invalid UTF-8
";
        assert_eq!(diagnostic.to_string(), expected);
    }

    #[test]
    fn highlight_eos_with_one_tilde() {
        let elaborated = Diagnostic {
            clauses: vec![Clause {
                file: DUMMY_FILE.to_string(),
                tag: Tag::Error,
                message: "unexpected end of file".into(),
                excerpt: Some(Excerpt {
                    line: LineNumber(2),
                    source: "dummy".to_string(),
                    highlight: mk_highlight(LineNumber(2), 5, 5),
                }),
            }],
        };
        let expected = r"/my/file:2: error: unexpected end of file
dummy
     ^
";
        assert_eq!(elaborated.to_string(), expected)
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
