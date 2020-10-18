use crate::codebase::{BufId, BufRange, TextCache};

use std::cmp::Ordering;
use std::ops::RangeInclusive;
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq)]
pub struct Spanned<T, S> {
    pub item: T,
    pub span: S,
}

pub trait WithSpan: Sized {
    fn with_span<S>(self, span: S) -> Spanned<Self, S> {
        Spanned { item: self, span }
    }
}

impl<T> WithSpan for T {}

pub trait SpanSource {
    type Span: Clone;
}

pub trait Source: SpanSource {
    fn span(&self) -> Self::Span;
}

pub(crate) trait MergeSpans<S> {
    fn merge_spans(&mut self, left: &S, right: &S) -> S;
}

pub(crate) trait StripSpan<S> {
    type Stripped;

    fn strip_span(&mut self, span: &S) -> Self::Stripped;
}

pub(crate) trait SpanSystem<F>
where
    Self:
        SpanSource + MergeSpans<<Self as SpanSource>::Span> + StripSpan<<Self as SpanSource>::Span>,
{
    type FileInclusionMetadataId: Clone;
    type MacroDefMetadataId: Clone + 'static;
    type MacroExpansionMetadataId: Clone + 'static;

    fn add_file_inclusion(
        &mut self,
        file_inclusion: FileInclusionMetadata<F, Self::Span>,
    ) -> Self::FileInclusionMetadataId;

    fn add_macro_def(
        &mut self,
        macro_def: MacroDefMetadata<Self::Span>,
    ) -> Self::MacroDefMetadataId;

    fn add_macro_expansion(
        &mut self,
        macro_expansion: MacroExpansionMetadata<Self::MacroDefMetadataId, Self::Span>,
    ) -> Self::MacroExpansionMetadataId;

    fn encode_span(
        &mut self,
        span: Span<Self::FileInclusionMetadataId, Self::MacroExpansionMetadataId>,
    ) -> Self::Span;
}

#[derive(Debug)]
pub struct FileInclusionMetadata<F, S> {
    pub file: F,
    pub from: Option<S>,
}

#[derive(Debug)]
pub struct MacroDefMetadata<S> {
    pub name_span: S,
    pub param_spans: Box<[S]>,
    pub body_spans: Box<[S]>,
}

#[derive(Debug)]
pub struct MacroExpansionMetadata<D, S> {
    pub def: D,
    pub name_span: S,
    pub arg_spans: Box<[Box<[S]>]>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Span<I, E> {
    File(I, BufRange),
    MacroExpansion(E, RangeInclusive<MacroExpansionPos>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MacroExpansionPos {
    pub token: usize,
    pub param_expansion: Option<ParamExpansionPos>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParamExpansionPos {
    pub param: usize,
    pub arg_token: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StrippedBufSpan<B, R> {
    pub buf_id: B,
    pub range: R,
}

impl TextCache {
    pub fn snippet(&self, stripped: &StrippedBufSpan<BufId, BufRange>) -> &str {
        &self.buf(stripped.buf_id).as_str()[stripped.range.clone()]
    }
}

impl PartialOrd for MacroExpansionPos {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.token != other.token {
            self.token.partial_cmp(&other.token)
        } else {
            match (&self.param_expansion, &other.param_expansion) {
                (Some(expansion), Some(other_expansion))
                    if expansion.param == other_expansion.param =>
                {
                    expansion.arg_token.partial_cmp(&other_expansion.arg_token)
                }
                (None, None) => Some(Ordering::Equal),
                _ => None,
            }
        }
    }
}

fn merge_macro_expansion_ranges(
    left: &RangeInclusive<MacroExpansionPos>,
    right: &RangeInclusive<MacroExpansionPos>,
) -> RangeInclusive<MacroExpansionPos> {
    assert!(left.start() <= right.end());
    left.start().clone()..=right.end().clone()
}

pub(crate) struct Spans<F> {
    source_file_inclusions:
        Vec<FileInclusionMetadata<F, Span<SourceFileInclusionId, MacroExpansionId>>>,
    macro_defs: Vec<MacroDefMetadata<Span<SourceFileInclusionId, MacroExpansionId>>>,
    pub macro_expansions:
        Vec<MacroExpansionMetadata<MacroDefId, Span<SourceFileInclusionId, MacroExpansionId>>>,
}

#[derive(Clone, Copy)]
pub struct MacroDefId(usize);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceFileInclusionId(usize);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacroExpansionId(pub usize);

impl<F> Default for Spans<F> {
    fn default() -> Self {
        Self {
            source_file_inclusions: Vec::default(),
            macro_defs: Vec::default(),
            macro_expansions: Vec::default(),
        }
    }
}

impl<F> SpanSource for Spans<F> {
    type Span = Span<SourceFileInclusionId, MacroExpansionId>;
}

#[derive(Debug)]
pub(crate) struct RcFileInclusion<F>(
    pub Rc<FileInclusionMetadata<F, Span<Self, RcMacroExpansion<F>>>>,
);

#[derive(Debug)]
pub(crate) struct RcMacroExpansion<F>(
    pub  Rc<
        MacroExpansionMetadata<
            Rc<MacroDefMetadata<Span<RcFileInclusion<F>, Self>>>,
            Span<RcFileInclusion<F>, Self>,
        >,
    >,
);

impl<F> Clone for RcFileInclusion<F> {
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}

impl<F> Clone for RcMacroExpansion<F> {
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}

impl<F> PartialEq for RcFileInclusion<F> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl<F> PartialEq for RcMacroExpansion<F> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl<F> MergeSpans<Span<SourceFileInclusionId, MacroExpansionId>> for Spans<F> {
    fn merge_spans(
        &mut self,
        left: &Span<SourceFileInclusionId, MacroExpansionId>,
        right: &Span<SourceFileInclusionId, MacroExpansionId>,
    ) -> Span<SourceFileInclusionId, MacroExpansionId> {
        match (left, right) {
            (
                Span::File(source_file_inclusion, range),
                Span::File(other_source_file_inclusion, other_range),
            ) if source_file_inclusion == other_source_file_inclusion => {
                Span::File(*source_file_inclusion, range.start..other_range.end)
            }
            (
                Span::MacroExpansion(macro_expansion, range),
                Span::MacroExpansion(other_macro_expansion, other_range),
            ) if macro_expansion == other_macro_expansion => Span::MacroExpansion(
                *macro_expansion,
                merge_macro_expansion_ranges(range, other_range),
            ),
            _ => unreachable!(),
        }
    }
}

impl<F: Clone> StripSpan<Span<SourceFileInclusionId, MacroExpansionId>> for Spans<F> {
    type Stripped = StrippedBufSpan<F, BufRange>;

    fn strip_span(
        &mut self,
        span: &Span<SourceFileInclusionId, MacroExpansionId>,
    ) -> Self::Stripped {
        match span {
            Span::File(SourceFileInclusionId(id), range) => StrippedBufSpan {
                buf_id: self.source_file_inclusions[*id].file.clone(),
                range: range.clone(),
            },
            Span::MacroExpansion(MacroExpansionId(id), range) => {
                let expansion = &self.macro_expansions[*id];
                let def = &self.macro_defs[expansion.def.0];
                let start = &def.body_spans[range.start().token];
                let end = &def.body_spans[range.end().token];
                let (buf_id, range) = match (start, end) {
                    (
                        Span::File(SourceFileInclusionId(id), range),
                        Span::File(SourceFileInclusionId(other_id), other_range),
                    ) if id == other_id => (
                        self.source_file_inclusions[*id].file.clone(),
                        range.start..other_range.end,
                    ),
                    _ => unimplemented!(),
                };
                StrippedBufSpan { buf_id, range }
            }
        }
    }
}

impl<F: Clone + 'static> SpanSystem<F> for Spans<F> {
    type FileInclusionMetadataId = SourceFileInclusionId;
    type MacroDefMetadataId = MacroDefId;
    type MacroExpansionMetadataId = MacroExpansionId;

    fn add_file_inclusion(
        &mut self,
        file_inclusion: FileInclusionMetadata<F, Self::Span>,
    ) -> Self::FileInclusionMetadataId {
        let id = SourceFileInclusionId(self.source_file_inclusions.len());
        self.source_file_inclusions.push(file_inclusion);
        id
    }

    fn add_macro_def(
        &mut self,
        macro_def: MacroDefMetadata<Self::Span>,
    ) -> Self::MacroDefMetadataId {
        let id = MacroDefId(self.macro_defs.len());
        self.macro_defs.push(macro_def);
        id
    }

    fn add_macro_expansion(
        &mut self,
        macro_expansion: MacroExpansionMetadata<Self::MacroDefMetadataId, Self::Span>,
    ) -> Self::MacroExpansionMetadataId {
        let id = MacroExpansionId(self.macro_expansions.len());
        self.macro_expansions.push(macro_expansion);
        id
    }

    fn encode_span(&mut self, span: Span<SourceFileInclusionId, MacroExpansionId>) -> Self::Span {
        span
    }
}

#[cfg(test)]
pub mod fake {
    use super::*;

    use crate::diagnostics::mock::Merge;

    use std::marker::PhantomData;

    pub struct FakeSpanSystem<F, S>(PhantomData<(F, S)>);

    impl<F, S> Default for FakeSpanSystem<F, S> {
        fn default() -> Self {
            Self(PhantomData)
        }
    }

    impl<F, S: Clone> SpanSource for FakeSpanSystem<F, S> {
        type Span = S;
    }

    impl<F, S: Clone + Merge> MergeSpans<S> for FakeSpanSystem<F, S> {
        fn merge_spans(&mut self, left: &S, right: &S) -> S {
            S::merge(left.clone(), right.clone())
        }
    }

    impl<F, S: Clone> StripSpan<S> for FakeSpanSystem<F, S> {
        type Stripped = S;

        fn strip_span(&mut self, span: &S) -> Self::Stripped {
            span.clone()
        }
    }

    impl<F, S: Clone + Default + Merge> SpanSystem<F> for FakeSpanSystem<F, S> {
        type FileInclusionMetadataId = ();
        type MacroDefMetadataId = ();
        type MacroExpansionMetadataId = ();

        fn add_file_inclusion(
            &mut self,
            _: FileInclusionMetadata<F, Self::Span>,
        ) -> Self::FileInclusionMetadataId {
        }

        fn add_macro_def(&mut self, _: MacroDefMetadata<Self::Span>) -> Self::MacroDefMetadataId {}

        fn add_macro_expansion(
            &mut self,
            _: MacroExpansionMetadata<Self::MacroDefMetadataId, Self::Span>,
        ) -> Self::MacroExpansionMetadataId {
        }

        fn encode_span(&mut self, _: Span<(), ()>) -> Self::Span {
            S::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::codebase::TextCache;

    use std::ops::Range;

    #[test]
    fn extend_span() {
        let mut codebase = TextCache::new();
        let src = "left right";
        let buf_id = codebase.add_src_buf("/my/file", src);
        let mut spans = Spans::default();
        let source_file_inclusion = spans.add_file_inclusion(FileInclusionMetadata {
            file: buf_id,
            from: None,
        });
        let left = spans.encode_span(Span::File(source_file_inclusion, 0..4));
        let right = spans.encode_span(Span::File(source_file_inclusion, 5..10));
        let combined = spans.merge_spans(&left, &right);
        assert_eq!(combined, Span::File(source_file_inclusion, 0..10))
    }

    #[test]
    fn merge_simple_macro_expansion_positions() {
        let start_pos = MacroExpansionPos {
            token: 2,
            param_expansion: None,
        };
        let start = start_pos.clone()..=start_pos.clone();
        let end_pos = MacroExpansionPos {
            token: 7,
            param_expansion: None,
        };
        let end = end_pos.clone()..=end_pos.clone();
        assert_eq!(
            merge_macro_expansion_ranges(&start, &end),
            start_pos..=end_pos
        )
    }

    #[test]
    fn strip_buf_span() {
        let file = 7;
        let range = 0..1;
        let mut spans = Spans::default();
        let source_file_inclusion =
            spans.add_file_inclusion(FileInclusionMetadata { file, from: None });
        let span = spans.encode_span(Span::File(source_file_inclusion, range.clone()));
        assert_eq!(
            spans.strip_span(&span),
            StrippedBufSpan {
                buf_id: file,
                range
            }
        )
    }

    #[test]
    fn strip_macro_span() {
        let buf_id = 1;
        let mut spans = Spans::default();
        let source_file_inclusion = spans.add_file_inclusion(FileInclusionMetadata {
            file: buf_id,
            from: None,
        });
        let position = MacroExpansionPos {
            token: 0,
            param_expansion: None,
        };
        let macro_base = 0;
        let def_name_span = spans.encode_span(Span::File(
            source_file_inclusion,
            macro_name_range(macro_base),
        ));
        let def_body_span = spans.encode_span(Span::File(
            source_file_inclusion,
            macro_body_range(macro_base),
        ));
        let macro_def = spans.add_macro_def(MacroDefMetadata {
            name_span: def_name_span,
            param_spans: Box::new([]),
            body_spans: Box::new([def_body_span]),
        });
        let expansion_name_span = spans.encode_span(Span::File(source_file_inclusion, 40..50));
        let macro_expansion = spans.add_macro_expansion(MacroExpansionMetadata {
            name_span: expansion_name_span,
            def: macro_def,
            arg_spans: Box::new([]),
        });
        let span = spans.encode_span(Span::MacroExpansion(
            macro_expansion,
            position.clone()..=position,
        ));
        let stripped = StrippedBufSpan {
            buf_id,
            range: macro_body_range(macro_base),
        };
        assert_eq!(spans.strip_span(&span), stripped)
    }

    fn macro_name_range(base: usize) -> Range<usize> {
        base..base + 10
    }

    fn macro_body_range(base: usize) -> Range<usize> {
        base + 20..base + 30
    }
}
