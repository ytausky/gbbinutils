use crate::codebase::{BufId, BufRange};
use std::cmp::Ordering;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::RangeInclusive;
use std::rc::Rc;

pub trait Span {
    type Span: Clone + Debug + PartialEq;
}

pub trait Source
where
    Self: Span,
{
    fn span(&self) -> Self::Span;
}

pub trait Merge
where
    Self: Span,
{
    fn merge(&mut self, left: &Self::Span, right: &Self::Span) -> Self::Span;
}

pub trait MacroContextFactory
where
    Self: Span,
{
    type MacroDefId: Clone;
    type MacroExpansionContext: MacroExpansionContext<Span = Self::Span>;

    fn add_macro_def<P, B>(&mut self, name: Self::Span, params: P, body: B) -> Self::MacroDefId
    where
        P: IntoIterator<Item = Self::Span>,
        B: IntoIterator<Item = Self::Span>;

    fn mk_macro_expansion_context<A, J>(
        &mut self,
        name: Self::Span,
        args: A,
        def: &Self::MacroDefId,
    ) -> Self::MacroExpansionContext
    where
        A: IntoIterator<Item = J>,
        J: IntoIterator<Item = Self::Span>;
}

pub trait ContextFactory
where
    Self: Merge,
    Self: MacroContextFactory,
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
    SpanData<B, R>: Clone + Debug + PartialEq,
{
    type Span = SpanData<B, R>;
}

impl<B, R> MacroContextFactory for RcContextFactory<B, R>
where
    SpanData<B, R>: Clone + Debug + PartialEq,
{
    type MacroDefId = Rc<MacroDef<Self::Span>>;
    type MacroExpansionContext = Rc<MacroExpansionData<SpanData<B, R>>>;

    fn add_macro_def<P, C>(&mut self, name: Self::Span, params: P, body: C) -> Self::MacroDefId
    where
        P: IntoIterator<Item = Self::Span>,
        C: IntoIterator<Item = Self::Span>,
    {
        Rc::new(MacroDef {
            name,
            params: params.into_iter().collect(),
            body: body.into_iter().collect(),
        })
    }

    fn mk_macro_expansion_context<I, J>(
        &mut self,
        name: Self::Span,
        args: I,
        def: &Self::MacroDefId,
    ) -> Self::MacroExpansionContext
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = Self::Span>,
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

impl Merge for RcContextFactory<BufId, BufRange> {
    fn merge(&mut self, left: &Self::Span, right: &Self::Span) -> Self::Span {
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
        let combined = RcContextFactory::new().merge(&left, &right);
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
