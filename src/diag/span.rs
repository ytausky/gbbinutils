use crate::codebase::{BufId, BufRange, TextCache};

use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::{Range, RangeInclusive};
use std::rc::Rc;

pub trait SpanSource {
    type Span: Clone;
}

pub trait Source: SpanSource {
    fn span(&self) -> Self::Span;
}

pub trait MergeSpans<S> {
    fn merge_spans(&mut self, left: &S, right: &S) -> S;
}

pub trait StripSpan<S> {
    type Stripped;

    fn strip_span(&mut self, span: &S) -> Self::Stripped;
}

pub trait MacroContextFactory<S: Clone> {
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

pub trait SpanSystem
where
    Self: BufContextFactory,
    Self: MacroContextFactory<<Self as SpanSource>::Span>,
    Self: MergeSpans<<Self as SpanSource>::Span>,
    Self: StripSpan<<Self as SpanSource>::Span>,
{
}

pub trait BufContextFactory: SpanSource {
    type BufContext: BufContext<Span = Self::Span>;

    fn mk_buf_context(
        &mut self,
        buf_id: BufId,
        included_from: Option<Self::Span>,
    ) -> Self::BufContext;
}

pub trait BufContext: SpanSource {
    fn mk_span(&self, range: BufRange) -> Self::Span;
}

pub trait MacroExpansionContext: SpanSource {
    fn mk_span(&self, position: MacroExpansionPos) -> Self::Span;
}

#[derive(Clone, Debug, PartialEq)]
pub struct MacroExpansionPos {
    pub token: usize,
    pub expansion: Option<ArgExpansionPos>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArgExpansionPos {
    pub arg: usize,
    pub token: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SpanData<B = BufSpan> {
    Buf(B),
    Macro {
        range: RangeInclusive<MacroExpansionPos>,
        context: Rc<MacroExpansionData<SpanData<B>>>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct BufSpan<B = BufId, R = BufRange> {
    pub range: R,
    pub context: Rc<BufContextData<B, R>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StrippedBufSpan<B = BufId, R = BufRange> {
    pub buf_id: B,
    pub range: R,
}

impl<B: Clone, T: Clone> SpanData<BufSpan<B, Range<T>>> {
    pub fn to_stripped(&self) -> StrippedBufSpan<B, Range<T>> {
        match self {
            SpanData::Buf(BufSpan { range, context }) => StrippedBufSpan {
                buf_id: context.buf_id.clone(),
                range: range.clone(),
            },
            SpanData::Macro { range, context } => {
                let start = &context.def.body[range.start().token];
                let end = &context.def.body[range.end().token];
                let (buf_id, range) = match (start, end) {
                    (
                        SpanData::Buf(BufSpan {
                            ref range,
                            ref context,
                        }),
                        SpanData::Buf(BufSpan {
                            range: ref other_range,
                            context: ref other_context,
                        }),
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

impl PartialOrd for MacroExpansionPos {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.token != other.token {
            self.token.partial_cmp(&other.token)
        } else {
            match (&self.expansion, &other.expansion) {
                (Some(expansion), Some(other_expansion))
                    if expansion.arg == other_expansion.arg =>
                {
                    expansion.token.partial_cmp(&other_expansion.token)
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
pub struct BufContextData<B, R> {
    pub buf_id: B,
    pub included_from: Option<SpanData<BufSpan<B, R>>>,
}

#[derive(Debug, PartialEq)]
pub struct MacroExpansionData<S> {
    pub name: S,
    pub args: Vec<Vec<S>>,
    pub def: Rc<MacroDefSpans<S>>,
}

#[derive(Debug, PartialEq)]
pub struct MacroDefSpans<S> {
    pub name: S,
    pub params: Vec<S>,
    pub body: Vec<S>,
}

pub struct RcContextFactory<B = BufId, R = BufRange>(PhantomData<(B, R)>);

impl<B, R> RcContextFactory<B, R> {
    pub fn new() -> Self {
        RcContextFactory(PhantomData)
    }
}

impl<B, R> SpanSource for RcContextFactory<B, R>
where
    SpanData<BufSpan<B, R>>: Clone,
{
    type Span = SpanData<BufSpan<B, R>>;
}

impl<B, R> MacroContextFactory<SpanData<BufSpan<B, R>>> for RcContextFactory<B, R>
where
    SpanData<BufSpan<B, R>>: Clone,
{
    type MacroDefId = Rc<MacroDefSpans<SpanData<BufSpan<B, R>>>>;
    type MacroExpansionContext = Rc<MacroExpansionData<SpanData<BufSpan<B, R>>>>;

    fn add_macro_def<P, C>(
        &mut self,
        name: SpanData<BufSpan<B, R>>,
        params: P,
        body: C,
    ) -> Self::MacroDefId
    where
        P: IntoIterator<Item = SpanData<BufSpan<B, R>>>,
        C: IntoIterator<Item = SpanData<BufSpan<B, R>>>,
    {
        Rc::new(MacroDefSpans {
            name,
            params: params.into_iter().collect(),
            body: body.into_iter().collect(),
        })
    }

    fn mk_macro_expansion_context<I, J>(
        &mut self,
        name: SpanData<BufSpan<B, R>>,
        args: I,
        def: &Self::MacroDefId,
    ) -> Self::MacroExpansionContext
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = SpanData<BufSpan<B, R>>>,
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

impl MergeSpans<SpanData<BufSpan>> for RcContextFactory<BufId, BufRange> {
    fn merge_spans(
        &mut self,
        left: &SpanData<BufSpan>,
        right: &SpanData<BufSpan>,
    ) -> SpanData<BufSpan> {
        use self::SpanData::*;
        match (left, right) {
            (
                Buf(BufSpan { range, context }),
                Buf(BufSpan {
                    range: other_range,
                    context: other_context,
                }),
            ) if Rc::ptr_eq(context, other_context) => Buf(BufSpan {
                range: range.start..other_range.end,
                context: Rc::clone(context),
            }),
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

impl StripSpan<SpanData<BufSpan>> for RcContextFactory<BufId, BufRange> {
    type Stripped = StrippedBufSpan<BufId, BufRange>;

    fn strip_span(&mut self, span: &SpanData<BufSpan>) -> Self::Stripped {
        span.to_stripped()
    }
}

impl SpanSystem for RcContextFactory<BufId, BufRange> {}

impl BufContextFactory for RcContextFactory<BufId, BufRange> {
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

impl SpanSource for RcBufContext<BufId, BufRange> {
    type Span = SpanData<BufSpan>;
}

impl BufContext for RcBufContext<BufId, BufRange> {
    fn mk_span(&self, range: BufRange) -> Self::Span {
        SpanData::Buf(BufSpan {
            range,
            context: self.context.clone(),
        })
    }
}

impl<B, R> SpanSource for Rc<MacroExpansionData<SpanData<BufSpan<B, R>>>>
where
    SpanData<BufSpan<B, R>>: Clone,
{
    type Span = SpanData<BufSpan<B, R>>;
}

impl<B, R> MacroExpansionContext for Rc<MacroExpansionData<SpanData<BufSpan<B, R>>>>
where
    SpanData<BufSpan<B, R>>: Clone,
{
    fn mk_span(&self, position: MacroExpansionPos) -> Self::Span {
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
        let left = SpanData::Buf(BufSpan {
            range: 0..4,
            context: context.clone(),
        });
        let right = SpanData::Buf(BufSpan {
            range: 5..10,
            context: context.clone(),
        });
        let combined = RcContextFactory::new().merge_spans(&left, &right);
        assert_eq!(
            combined,
            SpanData::Buf(BufSpan {
                range: 0..10,
                context
            })
        )
    }

    #[test]
    fn merge_simple_macro_expansion_positions() {
        let start_pos = MacroExpansionPos {
            token: 2,
            expansion: None,
        };
        let start = start_pos.clone()..=start_pos.clone();
        let end_pos = MacroExpansionPos {
            token: 7,
            expansion: None,
        };
        let end = end_pos.clone()..=end_pos.clone();
        assert_eq!(
            merge_macro_expansion_ranges(&start, &end),
            start_pos..=end_pos
        )
    }

    #[test]
    fn strip_buf_span() {
        let buf_id = 7;
        let range = 0..1;
        let context = Rc::new(BufContextData {
            buf_id,
            included_from: None,
        });
        let span = SpanData::Buf(BufSpan {
            range: range.clone(),
            context,
        });
        assert_eq!(span.to_stripped(), StrippedBufSpan { buf_id, range })
    }

    #[test]
    fn strip_macro_span() {
        let buf_id = 1;
        let buf_context = Rc::new(BufContextData {
            buf_id,
            included_from: None,
        });
        let position = MacroExpansionPos {
            token: 0,
            expansion: None,
        };
        let macro_base = 0;
        let expansion = MacroExpansionData {
            name: SpanData::Buf(BufSpan {
                range: 40..50,
                context: Rc::clone(&buf_context),
            }),
            args: vec![],
            def: mk_macro_def(&buf_context, macro_base),
        };
        let span = SpanData::Macro {
            range: position.clone()..=position,
            context: Rc::new(expansion),
        };
        let stripped = StrippedBufSpan {
            buf_id,
            range: macro_body_range(macro_base),
        };
        assert_eq!(span.to_stripped(), stripped)
    }

    type BufSpan<B> = super::BufSpan<B, Range<usize>>;

    fn mk_macro_def<B>(
        buf_context: &Rc<BufContextData<B, Range<usize>>>,
        base: usize,
    ) -> Rc<MacroDefSpans<SpanData<BufSpan<B>>>> {
        Rc::new(MacroDefSpans {
            name: SpanData::Buf(BufSpan {
                range: macro_name_range(base),
                context: Rc::clone(buf_context),
            }),
            params: vec![],
            body: vec![SpanData::Buf(BufSpan {
                range: macro_body_range(base),
                context: Rc::clone(buf_context),
            })],
        })
    }

    fn macro_name_range(base: usize) -> Range<usize> {
        base..base + 10
    }

    fn macro_body_range(base: usize) -> Range<usize> {
        base + 20..base + 30
    }
}
