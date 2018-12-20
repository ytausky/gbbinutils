use crate::codebase::{BufId, BufRange, TextCache};
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::{Range, RangeInclusive};
use std::rc::Rc;

pub trait Span {
    type Span: Clone;
}

pub trait Source: Span {
    fn span(&self) -> Self::Span;
}

pub trait MergeSpans<S> {
    fn merge_spans(&mut self, left: &S, right: &S) -> S;
}

pub trait StripSpan<S> {
    type Stripped;

    fn strip_span(&mut self, span: &S) -> Self::Stripped;
}

pub trait MacroContextFactory<S> {
    type MacroDefId: Clone;
    type MacroExpansionContext: MacroExpansionContext<Span = S>;

    fn add_macro_def<P, B>(&mut self, name: S, params: P, body: B) -> Self::MacroDefId
    where
        P: IntoIterator<Item = S>,
        B: IntoIterator<Item = S>;

    fn mk_macro_expansion_context<A, J>(
        &mut self,
        name: S,
        args: A,
        def: &Self::MacroDefId,
    ) -> Self::MacroExpansionContext
    where
        A: IntoIterator<Item = J>,
        J: IntoIterator<Item = S>;
}

pub trait ContextFactory
where
    Self: Span,
    Self: MacroContextFactory<<Self as Span>::Span>,
    Self: MergeSpans<<Self as Span>::Span>,
    Self: StripSpan<<Self as Span>::Span>,
{
    type BufContext: BufContext<Span = Self::Span>;

    fn mk_buf_context(
        &mut self,
        buf_id: BufId,
        included_from: Option<Self::Span>,
    ) -> Self::BufContext;
}

pub trait BufContext {
    type Span;
    fn mk_span(&self, range: BufRange) -> Self::Span;
}

pub trait MacroExpansionContext {
    type Span;
    fn mk_span(&self, token: usize, expansion: Option<TokenExpansion>) -> Self::Span;
}

#[derive(Clone, Debug, PartialEq)]
pub struct TokenExpansion {
    pub arg: usize,
    pub index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SpanData<B = BufId, R = BufRange> {
    Buf {
        range: R,
        context: Rc<BufContextData<B, R>>,
    },
    Macro {
        range: RangeInclusive<MacroExpansionPosition>,
        context: Rc<MacroExpansionData<SpanData<B, R>>>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct StrippedBufSpan<B = BufId, R = BufRange> {
    pub buf_id: B,
    pub range: R,
}

impl<B: Clone, T: Clone> SpanData<B, Range<T>> {
    pub fn to_stripped(&self) -> StrippedBufSpan<B, Range<T>> {
        match self {
            SpanData::Buf { range, context } => StrippedBufSpan {
                buf_id: context.buf_id.clone(),
                range: range.clone(),
            },
            SpanData::Macro { range, context } => {
                let start = &context.def.body[range.start().token];
                let end = &context.def.body[range.end().token];
                let (buf_id, range) = match (start, end) {
                    (
                        SpanData::Buf {
                            ref range,
                            ref context,
                        },
                        SpanData::Buf {
                            range: ref other_range,
                            context: ref other_context,
                        },
                    ) if Rc::ptr_eq(context, other_context) => (
                        context.buf_id.clone(),
                        range.start.clone()..other_range.end.clone(),
                    ),
                    _ => unimplemented!(),
                };
                StrippedBufSpan { buf_id, range }
            }
        }
    }
}

impl TextCache {
    pub fn snippet(&self, stripped: &StrippedBufSpan) -> &str {
        &self.buf(stripped.buf_id).as_str()[stripped.range.clone()]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MacroExpansionPosition {
    pub token: usize,
    pub expansion: Option<TokenExpansion>,
}

impl PartialOrd for MacroExpansionPosition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.token != other.token {
            self.token.partial_cmp(&other.token)
        } else {
            match (&self.expansion, &other.expansion) {
                (Some(expansion), Some(other_expansion))
                    if expansion.arg == other_expansion.arg =>
                {
                    expansion.index.partial_cmp(&other_expansion.index)
                }
                (None, None) => Some(Ordering::Equal),
                _ => None,
            }
        }
    }
}

fn merge_macro_expansion_ranges(
    left: &RangeInclusive<MacroExpansionPosition>,
    right: &RangeInclusive<MacroExpansionPosition>,
) -> RangeInclusive<MacroExpansionPosition> {
    assert!(left.start() <= right.end());
    left.start().clone()..=right.end().clone()
}

#[derive(Debug, PartialEq)]
pub struct BufContextData<B, R> {
    pub buf_id: B,
    pub included_from: Option<SpanData<B, R>>,
}

#[derive(Debug, PartialEq)]
pub struct MacroExpansionData<S> {
    pub name: S,
    pub args: Vec<Vec<S>>,
    pub def: Rc<MacroDef<S>>,
}

#[derive(Debug, PartialEq)]
pub struct MacroDef<S> {
    pub name: S,
    pub params: Vec<S>,
    pub body: Vec<S>,
}

pub struct RcContextFactory<B, R>(PhantomData<(B, R)>);

impl<B, R> RcContextFactory<B, R> {
    pub fn new() -> Self {
        RcContextFactory(PhantomData)
    }
}

impl<B, R> Span for RcContextFactory<B, R>
where
    SpanData<B, R>: Clone,
{
    type Span = SpanData<B, R>;
}

impl<B, R> MacroContextFactory<SpanData<B, R>> for RcContextFactory<B, R>
where
    SpanData<B, R>: Clone,
{
    type MacroDefId = Rc<MacroDef<SpanData<B, R>>>;
    type MacroExpansionContext = Rc<MacroExpansionData<SpanData<B, R>>>;

    fn add_macro_def<P, C>(&mut self, name: SpanData<B, R>, params: P, body: C) -> Self::MacroDefId
    where
        P: IntoIterator<Item = SpanData<B, R>>,
        C: IntoIterator<Item = SpanData<B, R>>,
    {
        Rc::new(MacroDef {
            name,
            params: params.into_iter().collect(),
            body: body.into_iter().collect(),
        })
    }

    fn mk_macro_expansion_context<I, J>(
        &mut self,
        name: SpanData<B, R>,
        args: I,
        def: &Self::MacroDefId,
    ) -> Self::MacroExpansionContext
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = SpanData<B, R>>,
    {
        Rc::new(MacroExpansionData {
            name,
            args: args
                .into_iter()
                .map(IntoIterator::into_iter)
                .map(Iterator::collect)
                .collect(),
            def: Rc::clone(def),
        })
    }
}

impl MergeSpans<SpanData<BufId, BufRange>> for RcContextFactory<BufId, BufRange> {
    fn merge_spans(
        &mut self,
        left: &SpanData<BufId, BufRange>,
        right: &SpanData<BufId, BufRange>,
    ) -> SpanData<BufId, BufRange> {
        use self::SpanData::*;
        match (left, right) {
            (
                Buf { range, context },
                Buf {
                    range: other_range,
                    context: other_context,
                },
            ) if Rc::ptr_eq(context, other_context) => Buf {
                range: range.start..other_range.end,
                context: Rc::clone(context),
            },
            (
                Macro { range, context },
                Macro {
                    range: other_range,
                    context: other_context,
                },
            ) if Rc::ptr_eq(context, other_context) => Macro {
                range: merge_macro_expansion_ranges(range, other_range),
                context: Rc::clone(context),
            },
            _ => unreachable!(),
        }
    }
}

impl StripSpan<SpanData<BufId, BufRange>> for RcContextFactory<BufId, BufRange> {
    type Stripped = StrippedBufSpan<BufId, BufRange>;

    fn strip_span(&mut self, span: &SpanData<BufId, BufRange>) -> Self::Stripped {
        span.to_stripped()
    }
}

impl ContextFactory for RcContextFactory<BufId, BufRange> {
    type BufContext = RcBufContext<BufId, BufRange>;

    fn mk_buf_context(
        &mut self,
        buf_id: BufId,
        included_from: Option<Self::Span>,
    ) -> Self::BufContext {
        let context = Rc::new(BufContextData {
            buf_id,
            included_from,
        });
        RcBufContext { context }
    }
}

#[derive(Clone)]
pub struct RcBufContext<B, R> {
    context: Rc<BufContextData<B, R>>,
}

impl BufContext for RcBufContext<BufId, BufRange> {
    type Span = SpanData<BufId, BufRange>;
    fn mk_span(&self, range: BufRange) -> Self::Span {
        SpanData::Buf {
            range,
            context: self.context.clone(),
        }
    }
}

impl<B, R> MacroExpansionContext for Rc<MacroExpansionData<SpanData<B, R>>> {
    type Span = SpanData<B, R>;

    fn mk_span(&self, token: usize, expansion: Option<TokenExpansion>) -> Self::Span {
        let position = MacroExpansionPosition { token, expansion };
        SpanData::Macro {
            range: position.clone()..=position,
            context: Rc::clone(self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebase::TextCache;

    #[test]
    fn extend_span() {
        let mut codebase = TextCache::new();
        let src = "left right";
        let buf_id = codebase.add_src_buf("/my/file", src);
        let context = Rc::new(BufContextData {
            buf_id,
            included_from: None,
        });
        let left = SpanData::Buf {
            range: BufRange::from(0..4),
            context: context.clone(),
        };
        let right = SpanData::Buf {
            range: BufRange::from(5..10),
            context: context.clone(),
        };
        let combined = RcContextFactory::new().merge_spans(&left, &right);
        assert_eq!(
            combined,
            SpanData::Buf {
                range: BufRange::from(0..10),
                context
            }
        )
    }

    #[test]
    fn merge_simple_macro_expansion_positions() {
        let start_pos = MacroExpansionPosition {
            token: 2,
            expansion: None,
        };
        let start = start_pos.clone()..=start_pos.clone();
        let end_pos = MacroExpansionPosition {
            token: 7,
            expansion: None,
        };
        let end = end_pos.clone()..=end_pos.clone();
        assert_eq!(
            merge_macro_expansion_ranges(&start, &end),
            start_pos..=end_pos
        )
    }
}
