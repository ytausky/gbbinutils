use crate::codebase::{BufId, BufRange, TextCache};

use std::cmp::Ordering;
use std::marker::PhantomData;
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

pub(crate) struct RcContextFactory<F>(PhantomData<F>);

impl<F> RcContextFactory<F> {
    pub fn new() -> Self {
        RcContextFactory(PhantomData)
    }
}

impl<F> SpanSource for RcContextFactory<F> {
    type Span = Span<RcFileInclusion<F>, RcMacroExpansion<F>>;
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

impl<F> MergeSpans<Span<RcFileInclusion<F>, RcMacroExpansion<F>>> for RcContextFactory<F> {
    fn merge_spans(
        &mut self,
        left: &Span<RcFileInclusion<F>, RcMacroExpansion<F>>,
        right: &Span<RcFileInclusion<F>, RcMacroExpansion<F>>,
    ) -> Span<RcFileInclusion<F>, RcMacroExpansion<F>> {
        match (left, right) {
            (
                Span::File(RcFileInclusion(file), range),
                Span::File(RcFileInclusion(other_file), other_range),
            ) if Rc::ptr_eq(file, other_file) => Span::File(
                RcFileInclusion(Rc::clone(file)),
                range.start..other_range.end,
            ),
            (
                Span::MacroExpansion(RcMacroExpansion(expansion), range),
                Span::MacroExpansion(RcMacroExpansion(other_expansion), other_range),
            ) if Rc::ptr_eq(expansion, other_expansion) => Span::MacroExpansion(
                RcMacroExpansion(Rc::clone(expansion)),
                merge_macro_expansion_ranges(range, other_range),
            ),
            _ => unreachable!(),
        }
    }
}

impl<F: Clone> StripSpan<Span<RcFileInclusion<F>, RcMacroExpansion<F>>> for RcContextFactory<F> {
    type Stripped = StrippedBufSpan<F, BufRange>;

    fn strip_span(
        &mut self,
        span: &Span<RcFileInclusion<F>, RcMacroExpansion<F>>,
    ) -> Self::Stripped {
        match span {
            Span::File(RcFileInclusion(file), range) => StrippedBufSpan {
                buf_id: file.file.clone(),
                range: range.clone(),
            },
            Span::MacroExpansion(RcMacroExpansion(expansion), range) => {
                let start = &expansion.def.body_spans[range.start().token];
                let end = &expansion.def.body_spans[range.end().token];
                let (buf_id, range) = match (start, end) {
                    (
                        Span::File(RcFileInclusion(file), range),
                        Span::File(RcFileInclusion(other_file), other_range),
                    ) if Rc::ptr_eq(file, other_file) => {
                        (file.file.clone(), range.start..other_range.end)
                    }
                    _ => unimplemented!(),
                };
                StrippedBufSpan { buf_id, range }
            }
        }
    }
}

impl<F: Clone + 'static> SpanSystem<F> for RcContextFactory<F> {
    type FileInclusionMetadataId = RcFileInclusion<F>;
    type MacroDefMetadataId = Rc<MacroDefMetadata<Self::Span>>;
    type MacroExpansionMetadataId = RcMacroExpansion<F>;

    fn add_file_inclusion(
        &mut self,
        file_inclusion: FileInclusionMetadata<F, Self::Span>,
    ) -> Self::FileInclusionMetadataId {
        RcFileInclusion(Rc::new(file_inclusion))
    }

    fn add_macro_def(
        &mut self,
        macro_def: MacroDefMetadata<Self::Span>,
    ) -> Self::MacroDefMetadataId {
        Rc::new(macro_def)
    }

    fn add_macro_expansion(
        &mut self,
        macro_expansion: MacroExpansionMetadata<Self::MacroDefMetadataId, Self::Span>,
    ) -> Self::MacroExpansionMetadataId {
        RcMacroExpansion(Rc::new(macro_expansion))
    }

    fn encode_span(&mut self, span: Span<RcFileInclusion<F>, RcMacroExpansion<F>>) -> Self::Span {
        span
    }
}

#[cfg(test)]
pub mod fake {
    use super::*;

    use crate::diagnostics::mock::Merge;

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
        let file = RcFileInclusion(Rc::new(FileInclusionMetadata {
            file: buf_id,
            from: None,
        }));
        let left = Span::<_, RcMacroExpansion<_>>::File(file.clone(), 0..4);
        let right = Span::<_, RcMacroExpansion<_>>::File(file.clone(), 5..10);
        let combined = RcContextFactory::new().merge_spans(&left, &right);
        assert_eq!(combined, Span::File(file, 0..10))
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
        let inclusion = Rc::new(FileInclusionMetadata { file, from: None });
        let span = Span::<_, RcMacroExpansion<_>>::File(RcFileInclusion(inclusion), range.clone());
        assert_eq!(
            RcContextFactory::new().strip_span(&span),
            StrippedBufSpan {
                buf_id: file,
                range
            }
        )
    }

    #[test]
    fn strip_macro_span() {
        let buf_id = 1;
        let buf_context = RcFileInclusion(Rc::new(FileInclusionMetadata {
            file: buf_id,
            from: None,
        }));
        let position = MacroExpansionPos {
            token: 0,
            param_expansion: None,
        };
        let macro_base = 0;
        let expansion = RcMacroExpansion(Rc::new(MacroExpansionMetadata {
            name_span: Span::File(buf_context.clone(), 40..50),
            def: mk_macro_def(&buf_context, macro_base),
            arg_spans: Box::new([]),
        }));
        let span = Span::MacroExpansion(expansion, position.clone()..=position);
        let stripped = StrippedBufSpan {
            buf_id,
            range: macro_body_range(macro_base),
        };
        assert_eq!(RcContextFactory::new().strip_span(&span), stripped)
    }

    fn mk_macro_def<F>(
        buf_context: &RcFileInclusion<F>,
        base: usize,
    ) -> Rc<MacroDefMetadata<Span<RcFileInclusion<F>, RcMacroExpansion<F>>>> {
        Rc::new(MacroDefMetadata {
            name_span: Span::File((*buf_context).clone(), macro_name_range(base)),
            param_spans: Box::new([]),
            body_spans: Box::new([Span::File((*buf_context).clone(), macro_body_range(base))]),
        })
    }

    fn macro_name_range(base: usize) -> Range<usize> {
        base..base + 10
    }

    fn macro_body_range(base: usize) -> Range<usize> {
        base + 20..base + 30
    }
}
