//! Types comprising the assembler's diagnostic output.
//!
//! During assembly the assembler may emit any number of diagnostic messages, each of which is
//! encoded in a [`Diagnostic`](struct.Diagnostic.html) along with all the information necessary to
//! present it to the end user.

pub(crate) use self::message::{KeywordOperandCategory, Message};
pub use crate::codebase::{LineNumber, TextPosition, TextRange};

use crate::codebase::{BufId, BufRange, TextBuf, TextCache};
use crate::span::*;

use std::cell::RefCell;
#[cfg(test)]
use std::marker::PhantomData;
use std::ops::Range;

mod message;
pub(crate) mod span;

#[cfg(test)]
pub(crate) use self::mock::*;

pub(crate) trait Diagnostics
where
    Self: Span,
    Self: ContextFactory,
    Self: DownstreamDiagnostics<<Self as Span>::Span>,
{
}

pub(crate) trait DownstreamDiagnostics<S>
where
    Self: MergeSpans<S>,
    Self: BackendDiagnostics<S>,
{
}

pub(crate) trait BackendDiagnostics<S>
where
    Self: StripSpan<S>,
    Self: EmitDiagnostic<S, <Self as StripSpan<S>>::Stripped>,
{
}

impl<T, S> BackendDiagnostics<S> for T
where
    T: StripSpan<S>,
    T: EmitDiagnostic<S, <T as StripSpan<S>>::Stripped>,
{
}

impl<T, S> DownstreamDiagnostics<S> for T
where
    T: MergeSpans<S>,
    T: BackendDiagnostics<S>,
{
}

pub(crate) trait DelegateDiagnostics<S> {
    type Delegate: DownstreamDiagnostics<S>;

    fn diagnostics(&mut self) -> &mut Self::Delegate;
}

pub(crate) struct DiagnosticsSystem<C, O> {
    pub context: C,
    pub output: O,
}

impl<'a> DiagnosticsSystem<RcContextFactory, OutputForwarder<'a>> {
    pub fn new(codebase: &'a RefCell<TextCache>, output: &'a mut dyn FnMut(Diagnostic)) -> Self {
        DiagnosticsSystem {
            context: RcContextFactory::new(),
            output: OutputForwarder { output, codebase },
        }
    }
}

impl<C, O> Span for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: EmitDiagnostic<C::Span, C::Stripped>,
{
    type Span = C::Span;
}

impl<C, O> MergeSpans<C::Span> for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: EmitDiagnostic<C::Span, C::Stripped>,
{
    fn merge_spans(&mut self, left: &C::Span, right: &C::Span) -> C::Span {
        self.context.merge_spans(left, right)
    }
}

impl<C, O> StripSpan<C::Span> for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: EmitDiagnostic<C::Span, C::Stripped>,
{
    type Stripped = C::Stripped;

    fn strip_span(&mut self, span: &C::Span) -> Self::Stripped {
        self.context.strip_span(span)
    }
}

impl<C, O> EmitDiagnostic<C::Span, C::Stripped> for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: EmitDiagnostic<C::Span, C::Stripped>,
{
    fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<C::Span, C::Stripped>>) {
        self.output.emit_diagnostic(diagnostic)
    }
}

impl<C, O> MacroContextFactory<C::Span> for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: EmitDiagnostic<C::Span, C::Stripped>,
{
    type MacroDefId = C::MacroDefId;
    type MacroExpansionContext = C::MacroExpansionContext;

    fn add_macro_def<P, B>(&mut self, name: C::Span, params: P, body: B) -> Self::MacroDefId
    where
        P: IntoIterator<Item = C::Span>,
        B: IntoIterator<Item = C::Span>,
    {
        self.context.add_macro_def(name, params, body)
    }

    fn mk_macro_expansion_context<A, J>(
        &mut self,
        name: C::Span,
        args: A,
        def: &Self::MacroDefId,
    ) -> Self::MacroExpansionContext
    where
        A: IntoIterator<Item = J>,
        J: IntoIterator<Item = C::Span>,
    {
        self.context.mk_macro_expansion_context(name, args, def)
    }
}

impl<C, O> ContextFactory for DiagnosticsSystem<C, O>
where
    C: ContextFactory,
    O: EmitDiagnostic<C::Span, C::Stripped>,
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
    O: EmitDiagnostic<C::Span, C::Stripped>,
{
}

pub trait DiagnosticsOutput {
    fn emit(&mut self, diagnostic: Diagnostic);
}

pub(crate) trait EmitDiagnostic<S, T> {
    fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<S, T>>);
}

pub(crate) struct OutputForwarder<'a> {
    pub output: &'a mut dyn FnMut(Diagnostic),
    pub codebase: &'a RefCell<TextCache>,
}

impl<'a> Span for OutputForwarder<'a> {
    type Span = SpanData;
}

impl<'a> EmitDiagnostic<SpanData, StrippedBufSpan> for OutputForwarder<'a> {
    fn emit_diagnostic(
        &mut self,
        diagnostic: impl Into<CompactDiagnostic<SpanData, StrippedBufSpan>>,
    ) {
        (self.output)(diagnostic.into().expand().render(&self.codebase.borrow()))
    }
}

#[cfg(test)]
pub(crate) struct IgnoreDiagnostics<S>(PhantomData<S>);

#[cfg(test)]
impl<S> IgnoreDiagnostics<S> {
    pub fn new() -> Self {
        IgnoreDiagnostics(PhantomData)
    }
}

#[cfg(test)]
impl<S: Clone> Span for IgnoreDiagnostics<S> {
    type Span = S;
}

#[cfg(test)]
impl<S: Clone> StripSpan<S> for IgnoreDiagnostics<S> {
    type Stripped = S;

    fn strip_span(&mut self, span: &S) -> Self::Stripped {
        span.clone()
    }
}

#[cfg(test)]
impl<S: Clone> EmitDiagnostic<S, S> for IgnoreDiagnostics<S> {
    fn emit_diagnostic(&mut self, _: impl Into<CompactDiagnostic<S, S>>) {}
}

#[cfg(test)]
pub(crate) struct TestDiagnosticsListener<S> {
    pub diagnostics: RefCell<Vec<CompactDiagnostic<S, S>>>,
}

#[cfg(test)]
impl<S> TestDiagnosticsListener<S> {
    pub fn new() -> TestDiagnosticsListener<S> {
        TestDiagnosticsListener {
            diagnostics: RefCell::new(Vec::new()),
        }
    }
}

#[cfg(test)]
impl<S: Clone> Span for TestDiagnosticsListener<S> {
    type Span = S;
}

#[cfg(test)]
impl<S: Clone> StripSpan<S> for TestDiagnosticsListener<S> {
    type Stripped = S;

    fn strip_span(&mut self, span: &S) -> Self::Stripped {
        span.clone()
    }
}

#[cfg(test)]
impl<S> EmitDiagnostic<S, S> for TestDiagnosticsListener<S> {
    fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<S, S>>) {
        self.diagnostics.borrow_mut().push(diagnostic.into())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompactDiagnostic<S, R> {
    main: CompactClause<S, R>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompactClause<S, R> {
    pub message: Message<R>,
    pub highlight: S,
}

impl<S, R> From<CompactClause<S, R>> for CompactDiagnostic<S, R> {
    fn from(clause: CompactClause<S, R>) -> Self {
        Self { main: clause }
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
struct ExpandedDiagnostic<S, B, R> {
    clauses: Vec<ExpandedDiagnosticClause<S, B, R>>,
}

#[derive(Debug, PartialEq)]
struct ExpandedDiagnosticClause<S, B, R> {
    buf_id: B,
    tag: Tag,
    message: Message<S>,
    location: Option<R>,
}

impl<B: Clone, T: Clone> CompactDiagnostic<SpanData<B, Range<T>>, StrippedBufSpan<B, Range<T>>> {
    fn expand(self) -> ExpandedDiagnostic<StrippedBufSpan<B, Range<T>>, B, Range<T>> {
        let StrippedBufSpan { buf_id, range } = self.main.highlight.to_stripped();
        let main_clause = ExpandedDiagnosticClause {
            buf_id,
            tag: Tag::Error,
            message: self.main.message,
            location: Some(range),
        };
        let mut clauses = vec![main_clause];
        if let Some(note) = mk_called_here_clause(&self.main.highlight) {
            clauses.push(note)
        }
        ExpandedDiagnostic { clauses }
    }
}

type BufSnippetClause<B, T> = ExpandedDiagnosticClause<StrippedBufSpan<B, Range<T>>, B, Range<T>>;

fn mk_called_here_clause<B: Clone, T: Clone>(
    span: &SpanData<B, Range<T>>,
) -> Option<BufSnippetClause<B, T>> {
    let call = if let SpanData::Macro { context, .. } = span {
        context.name.clone()
    } else {
        return None;
    };
    let stripped = call.to_stripped();
    Some(ExpandedDiagnosticClause {
        buf_id: stripped.buf_id.clone(),
        tag: Tag::Note,
        location: Some(stripped.range.clone()),
        message: Message::CalledHere { name: stripped },
    })
}

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

pub(crate) fn mk_diagnostic(
    file: impl Into<String>,
    message: &Message<StrippedBufSpan>,
) -> Diagnostic {
    Diagnostic {
        clauses: vec![Clause {
            file: file.into(),
            tag: Tag::Error,
            message: message.render(&TextCache::new()),
            excerpt: None,
        }],
    }
}

impl ExpandedDiagnostic<StrippedBufSpan, BufId, BufRange> {
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

impl ExpandedDiagnosticClause<StrippedBufSpan, BufId, BufRange> {
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
mod mock {
    use super::*;

    pub(crate) trait MockSpan {
        fn default() -> Self;
        fn merge(&self, other: &Self) -> Self;
    }

    impl MockSpan for () {
        fn default() {}
        fn merge(&self, _: &Self) {}
    }

    pub(crate) struct MockDiagnostics<'a, T, S> {
        log: &'a RefCell<Vec<T>>,
        _span: PhantomData<S>,
    }

    impl<'a, T, S> MockDiagnostics<'a, T, S> {
        pub fn new(log: &'a RefCell<Vec<T>>) -> Self {
            MockDiagnostics {
                log,
                _span: PhantomData,
            }
        }
    }

    impl<'a, T, S> Clone for MockDiagnostics<'a, T, S> {
        fn clone(&self) -> Self {
            MockDiagnostics {
                log: self.log,
                _span: PhantomData,
            }
        }
    }

    #[derive(Debug, PartialEq)]
    pub(crate) enum DiagnosticsEvent<S> {
        EmitDiagnostic(CompactDiagnostic<S, S>),
    }

    impl<S> From<CompactDiagnostic<S, S>> for DiagnosticsEvent<S> {
        fn from(diagnostic: CompactDiagnostic<S, S>) -> Self {
            DiagnosticsEvent::EmitDiagnostic(diagnostic)
        }
    }

    impl<'a, T, S> Diagnostics for MockDiagnostics<'a, T, S>
    where
        T: From<DiagnosticsEvent<S>>,
        S: Clone + MockSpan,
    {
    }

    impl<'a, T, S> ContextFactory for MockDiagnostics<'a, T, S>
    where
        S: Clone + MockSpan,
    {
        type BufContext = Self;

        fn mk_buf_context(&mut self, _: BufId, _: Option<Self::Span>) -> Self::BufContext {
            self.clone()
        }
    }

    impl<'a, T, S: Clone + MockSpan> BufContext for MockDiagnostics<'a, T, S> {
        type Span = S;

        fn mk_span(&self, _: BufRange) -> Self::Span {
            S::default()
        }
    }

    impl<'a, T, S> MacroContextFactory<S> for MockDiagnostics<'a, T, S>
    where
        S: Clone + MockSpan,
    {
        type MacroDefId = usize;
        type MacroExpansionContext = Self;

        fn add_macro_def<P, B>(&mut self, _: S, _: P, _: B) -> Self::MacroDefId
        where
            P: IntoIterator<Item = S>,
            B: IntoIterator<Item = S>,
        {
            0
        }

        fn mk_macro_expansion_context<A, J>(
            &mut self,
            _: S,
            _: A,
            _: &Self::MacroDefId,
        ) -> Self::MacroExpansionContext
        where
            A: IntoIterator<Item = J>,
            J: IntoIterator<Item = S>,
        {
            self.clone()
        }
    }

    impl<'a, T, S: Clone + MockSpan> MacroExpansionContext for MockDiagnostics<'a, T, S> {
        type Span = S;

        fn mk_span(&self, _: usize, _: Option<TokenExpansion>) -> Self::Span {
            S::default()
        }
    }

    impl<'a, T, S: Clone> Span for MockDiagnostics<'a, T, S> {
        type Span = S;
    }

    impl<'a, T, S: Clone> StripSpan<S> for MockDiagnostics<'a, T, S> {
        type Stripped = S;

        fn strip_span(&mut self, span: &S) -> Self::Stripped {
            span.clone()
        }
    }

    impl<'a, T, S: Clone + MockSpan> MergeSpans<S> for MockDiagnostics<'a, T, S> {
        fn merge_spans(&mut self, left: &S, right: &S) -> S {
            left.merge(right)
        }
    }

    impl<'a, T, S> EmitDiagnostic<S, S> for MockDiagnostics<'a, T, S>
    where
        T: From<DiagnosticsEvent<S>>,
        S: Clone,
    {
        fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<S, S>>) {
            self.log
                .borrow_mut()
                .push(DiagnosticsEvent::EmitDiagnostic(diagnostic.into()).into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebase::TextPosition;
    use crate::span::BufContextData;
    use std::rc::Rc;

    static DUMMY_FILE: &str = "/my/file";

    #[test]
    fn mk_message_for_undefined_macro() {
        let mut codebase = TextCache::new();
        let src = "    nop\n    my_macro a, $12\n\n";
        let buf_id = codebase.add_src_buf(DUMMY_FILE, src);
        let range = 12..20;
        let token_ref = SpanData::Buf {
            range: range.clone(),
            context: Rc::new(BufContextData {
                buf_id,
                included_from: None,
            }),
        };
        let diagnostic = CompactDiagnostic::from(
            Message::UndefinedMacro {
                name: StrippedBufSpan { buf_id, range },
            }
            .at(token_ref),
        );
        assert_eq!(
            diagnostic.expand().render(&codebase),
            Diagnostic {
                clauses: vec![Clause {
                    file: DUMMY_FILE.to_string(),
                    tag: Tag::Error,
                    message: "invocation of undefined macro `my_macro`".to_string(),
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
        let buf_context = &Rc::new(BufContextData {
            buf_id: (),
            included_from: None,
        });
        let macro_def = Rc::new(MacroDef {
            name: SpanData::Buf {
                range: 0..1,
                context: Rc::clone(buf_context),
            },
            params: vec![],
            body: vec![SpanData::Buf {
                range: 2..3,
                context: Rc::clone(buf_context),
            }],
        });
        let call_range = 10..11;
        let context = Rc::new(MacroExpansionData {
            name: SpanData::Buf {
                range: call_range.clone(),
                context: Rc::clone(buf_context),
            },
            args: vec![],
            def: macro_def,
        });
        let position = MacroExpansionPosition {
            token: 0,
            expansion: None,
        };
        let span = SpanData::Macro {
            range: position.clone()..=position,
            context,
        };
        let message = Message::AfOutsideStackOperation;
        let compact = CompactDiagnostic::from(message.clone().at(span));
        let expected = ExpandedDiagnostic {
            clauses: vec![
                ExpandedDiagnosticClause {
                    buf_id: (),
                    tag: Tag::Error,
                    message,
                    location: Some(2..3),
                },
                ExpandedDiagnosticClause {
                    buf_id: (),
                    tag: Tag::Note,
                    message: Message::CalledHere {
                        name: StrippedBufSpan {
                            buf_id: (),
                            range: call_range,
                        },
                    },
                    location: Some(10..11),
                },
            ],
        };
        assert_eq!(compact.expand(), expected)
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
