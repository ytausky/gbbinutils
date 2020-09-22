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

pub(crate) trait SpanSystem<T, R>
where
    Self:
        SpanSource + MergeSpans<<Self as SpanSource>::Span> + StripSpan<<Self as SpanSource>::Span>,
{
    fn encode_span(
        &mut self,
        span: Span<Rc<FileInclusion<BufId, Self::Span>>, Rc<MacroExpansion<T, R, Self::Span>>>,
    ) -> Self::Span;
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Span<I, E> {
    File(I, BufRange),
    MacroExpansion(E, RangeInclusive<MacroExpansionPos>),
}

#[derive(Debug)]
pub(crate) struct FileInclusion<F, S> {
    pub file: F,
    pub from: Option<S>,
}

#[derive(Debug)]
pub(crate) struct MacroExpansion<T, R, S> {
    pub name_span: S,
    pub def: Rc<MacroDef<T, R, S>>,
    pub args: Box<[Box<[(T, S)]>]>,
}

#[derive(Debug)]
pub(crate) struct MacroDef<T, R, S> {
    pub name_span: S,
    pub params: Box<[(R, S)]>,
    pub body: Box<[(T, S)]>,
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

pub(crate) struct RcContextFactory<F, T, R>(PhantomData<(F, T, R)>);

impl<F, T, R> RcContextFactory<F, T, R> {
    pub fn new() -> Self {
        RcContextFactory(PhantomData)
    }
}

impl<F: Clone, T, R> SpanSource for RcContextFactory<F, T, R> {
    type Span = Span<RcFileInclusion<F, T, R>, RcMacroExpansion<F, T, R>>;
}

#[derive(Debug)]
pub(crate) struct RcFileInclusion<F, T, R>(
    pub Rc<FileInclusion<F, Span<Self, RcMacroExpansion<F, T, R>>>>,
);

#[derive(Debug)]
pub(crate) struct RcMacroExpansion<F, T, R>(
    pub Rc<MacroExpansion<T, R, Span<RcFileInclusion<F, T, R>, Self>>>,
);

impl<F, T, R> Clone for RcFileInclusion<F, T, R> {
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}

impl<F, T, R> Clone for RcMacroExpansion<F, T, R> {
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}

impl<F, T, R> PartialEq for RcFileInclusion<F, T, R> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl<F, T, R> PartialEq for RcMacroExpansion<F, T, R> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl<F, T, R> MergeSpans<Span<RcFileInclusion<F, T, R>, RcMacroExpansion<F, T, R>>>
    for RcContextFactory<F, T, R>
{
    fn merge_spans(
        &mut self,
        left: &Span<RcFileInclusion<F, T, R>, RcMacroExpansion<F, T, R>>,
        right: &Span<RcFileInclusion<F, T, R>, RcMacroExpansion<F, T, R>>,
    ) -> Span<RcFileInclusion<F, T, R>, RcMacroExpansion<F, T, R>> {
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

impl<F: Clone, T, R> StripSpan<Span<RcFileInclusion<F, T, R>, RcMacroExpansion<F, T, R>>>
    for RcContextFactory<F, T, R>
{
    type Stripped = StrippedBufSpan<F, BufRange>;

    fn strip_span(
        &mut self,
        span: &Span<RcFileInclusion<F, T, R>, RcMacroExpansion<F, T, R>>,
    ) -> Self::Stripped {
        match span {
            Span::File(RcFileInclusion(file), range) => StrippedBufSpan {
                buf_id: file.file.clone(),
                range: range.clone(),
            },
            Span::MacroExpansion(RcMacroExpansion(expansion), range) => {
                let start = &expansion.def.body[range.start().token].1;
                let end = &expansion.def.body[range.end().token].1;
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

impl<T, R> SpanSystem<T, R> for RcContextFactory<BufId, T, R> {
    fn encode_span(
        &mut self,
        span: Span<Rc<FileInclusion<BufId, Self::Span>>, Rc<MacroExpansion<T, R, Self::Span>>>,
    ) -> Self::Span {
        match span {
            Span::File(file, range) => Span::File(RcFileInclusion(file), range),
            Span::MacroExpansion(expansion, range) => {
                Span::MacroExpansion(RcMacroExpansion(expansion), range)
            }
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
        let file = RcFileInclusion::<_, (), ()>(Rc::new(FileInclusion {
            file: buf_id,
            from: None,
        }));
        let left = Span::<_, RcMacroExpansion<_, _, _>>::File(file.clone(), 0..4);
        let right = Span::<_, RcMacroExpansion<_, _, _>>::File(file.clone(), 5..10);
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
        let inclusion = Rc::new(FileInclusion { file, from: None });
        let span = Span::<_, RcMacroExpansion<_, (), ()>>::File(
            RcFileInclusion(Rc::clone(&inclusion)),
            range.clone(),
        );
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
        let buf_context = RcFileInclusion(Rc::new(FileInclusion {
            file: buf_id,
            from: None,
        }));
        let position = MacroExpansionPos {
            token: 0,
            param_expansion: None,
        };
        let macro_base = 0;
        let expansion = RcMacroExpansion(Rc::new(MacroExpansion {
            name_span: Span::File(buf_context.clone(), 40..50),
            def: mk_macro_def(&buf_context, macro_base),
            args: Box::new([]),
        }));
        let span = Span::MacroExpansion(expansion, position.clone()..=position);
        let stripped = StrippedBufSpan {
            buf_id,
            range: macro_body_range(macro_base),
        };
        assert_eq!(RcContextFactory::new().strip_span(&span), stripped)
    }

    fn mk_macro_def<B>(
        buf_context: &RcFileInclusion<B, (), ()>,
        base: usize,
    ) -> Rc<MacroDef<(), (), Span<RcFileInclusion<B, (), ()>, RcMacroExpansion<B, (), ()>>>> {
        Rc::new(MacroDef {
            name_span: Span::File((*buf_context).clone(), macro_name_range(base)),
            params: Box::new([]),
            body: Box::new([(
                (),
                Span::File((*buf_context).clone(), macro_body_range(base)),
            )]),
        })
    }

    fn macro_name_range(base: usize) -> Range<usize> {
        base..base + 10
    }

    fn macro_body_range(base: usize) -> Range<usize> {
        base + 20..base + 30
    }
}
