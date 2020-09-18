use crate::codebase::{BufId, BufRange, TextCache};
use crate::session::lex::Literal;
use crate::syntax::Token;

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

pub(crate) trait SpanSystem<I, R>
where
    Self:
        SpanSource + MergeSpans<<Self as SpanSource>::Span> + StripSpan<<Self as SpanSource>::Span>,
{
    fn encode_span(&mut self, draft: SpanDraft<BufId, I, R, Self::Span>) -> Self::Span;
}

pub(crate) enum SpanDraft<F, I, R, S> {
    File(Rc<FileInclusion<F, S>>, BufRange),
    MacroExpansion(
        Rc<MacroExpansion<I, R, S>>,
        RangeInclusive<MacroExpansionPos>,
    ),
}

#[derive(Debug)]
pub(crate) struct FileInclusion<F, S> {
    pub file: F,
    pub from: Option<S>,
}

#[derive(Debug)]
pub(crate) struct MacroExpansion<I, R, S> {
    pub name_span: S,
    pub def: Rc<MacroDef<I, R, S>>,
    pub args: Box<[Box<[(Token<I, Literal<R>>, S)]>]>,
}

#[derive(Debug)]
pub(crate) struct MacroDef<I, R, S> {
    pub name_span: S,
    pub params: Box<[(I, S)]>,
    pub body: Box<[(Token<I, Literal<R>>, S)]>,
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

#[derive(Debug, PartialEq)]
pub struct ModularMacroCall<D, S> {
    pub name: S,
    pub args: Vec<Vec<S>>,
    pub def: D,
}

#[derive(Debug, PartialEq)]
pub struct MacroDefSpans<S> {
    pub name: S,
    pub params: Vec<S>,
    pub body: Vec<S>,
}

pub(crate) struct RcContextFactory<F, I, R>(PhantomData<(F, I, R)>);

impl<F, I, R> RcContextFactory<F, I, R> {
    pub fn new() -> Self {
        RcContextFactory(PhantomData)
    }
}

impl<F: Clone, I, R> SpanSource for RcContextFactory<F, I, R> {
    type Span = Span<F, I, R>;
}

impl<F, I, R> MergeSpans<Span<F, I, R>> for RcContextFactory<F, I, R> {
    fn merge_spans(&mut self, left: &Span<F, I, R>, right: &Span<F, I, R>) -> Span<F, I, R> {
        match (left, right) {
            (Span::File(file, range), Span::File(other_file, other_range))
                if Rc::ptr_eq(file, other_file) =>
            {
                Span::File(Rc::clone(file), range.start..other_range.end)
            }
            (
                Span::MacroExpansion(expansion, range),
                Span::MacroExpansion(other_expansion, other_range),
            ) if Rc::ptr_eq(expansion, other_expansion) => Span::MacroExpansion(
                Rc::clone(expansion),
                merge_macro_expansion_ranges(range, other_range),
            ),
            _ => unreachable!(),
        }
    }
}

impl<F: Clone, I, R> StripSpan<Span<F, I, R>> for RcContextFactory<F, I, R> {
    type Stripped = StrippedBufSpan<F, BufRange>;

    fn strip_span(&mut self, span: &Span<F, I, R>) -> Self::Stripped {
        match span {
            Span::File(file, range) => StrippedBufSpan {
                buf_id: file.file.clone(),
                range: range.clone(),
            },
            Span::MacroExpansion(expansion, range) => {
                let start = &expansion.def.body[range.start().token].1;
                let end = &expansion.def.body[range.end().token].1;
                let (buf_id, range) = match (start, end) {
                    (Span::File(file, range), Span::File(other_file, other_range))
                        if Rc::ptr_eq(file, other_file) =>
                    {
                        (
                            file.file.clone(),
                            range.start.clone()..other_range.end.clone(),
                        )
                    }
                    _ => unimplemented!(),
                };
                StrippedBufSpan { buf_id, range }
            }
        }
    }
}

impl<I, R> SpanSystem<I, R> for RcContextFactory<BufId, I, R> {
    fn encode_span(&mut self, draft: SpanDraft<BufId, I, R, Self::Span>) -> Self::Span {
        match draft {
            SpanDraft::File(file, range) => Span::File(file, range),
            SpanDraft::MacroExpansion(expansion, range) => Span::MacroExpansion(expansion, range),
        }
    }
}

#[derive(Debug)]
pub(crate) enum Span<F, I, R> {
    File(Rc<FileInclusion<F, Self>>, BufRange),
    MacroExpansion(
        Rc<MacroExpansion<I, R, Self>>,
        RangeInclusive<MacroExpansionPos>,
    ),
}

impl<F, I, R> Clone for Span<F, I, R> {
    fn clone(&self) -> Self {
        match self {
            Self::File(file, range) => Self::File(Rc::clone(file), range.clone()),
            Self::MacroExpansion(expansion, range) => {
                Self::MacroExpansion(Rc::clone(expansion), range.clone())
            }
        }
    }
}

impl<F, I, R> PartialEq for Span<F, I, R> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::File(inclusion, range), Self::File(other_inclusion, other_range)) => {
                Rc::ptr_eq(inclusion, other_inclusion) && range == other_range
            }
            (
                Self::MacroExpansion(expansion, range),
                Self::MacroExpansion(other_expansion, other_range),
            ) => Rc::ptr_eq(expansion, other_expansion) && range == other_range,
            _ => false,
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
        let file = Rc::new(FileInclusion {
            file: buf_id,
            from: None,
        });
        let left = Span::<_, (), ()>::File(Rc::clone(&file), 0..4);
        let right = Span::<_, (), ()>::File(Rc::clone(&file), 5..10);
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
        let span = Span::<_, (), ()>::File(Rc::clone(&inclusion), range.clone());
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
        let buf_context = Rc::new(FileInclusion {
            file: buf_id,
            from: None,
        });
        let position = MacroExpansionPos {
            token: 0,
            param_expansion: None,
        };
        let macro_base = 0;
        let expansion = Rc::new(MacroExpansion {
            name_span: Span::File(Rc::clone(&buf_context), 40..50),
            def: mk_macro_def(&buf_context, macro_base),
            args: Box::new([]),
        });
        let span = Span::MacroExpansion(expansion, position.clone()..=position);
        let stripped = StrippedBufSpan {
            buf_id,
            range: macro_body_range(macro_base),
        };
        assert_eq!(RcContextFactory::new().strip_span(&span), stripped)
    }

    fn mk_macro_def<B>(
        buf_context: &Rc<FileInclusion<B, Span<B, (), ()>>>,
        base: usize,
    ) -> Rc<MacroDef<(), (), Span<B, (), ()>>> {
        Rc::new(MacroDef {
            name_span: Span::File(Rc::clone(buf_context), macro_name_range(base)),
            params: Box::new([]),
            body: Box::new([(
                Token::Ident(()),
                Span::File(Rc::clone(buf_context), macro_body_range(base)),
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
