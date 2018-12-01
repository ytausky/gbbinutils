use crate::codebase::{BufId, BufRange};
use std::cmp;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;

pub trait Span: Clone + Debug + PartialEq {
    fn extend(&self, other: &Self) -> Self;
}

#[cfg(test)]
impl Span for () {
    fn extend(&self, _: &Self) -> Self {}
}

pub trait Source {
    type Span: Span;
    fn span(&self) -> Self::Span;
}

pub trait ContextFactory<I = BufId, R = BufRange> {
    type Span: Span;
    type MacroDefId: Clone;
    type BufContext: BufContext<R, Span = Self::Span>;
    type MacroExpansionContext: MacroExpansionContext<Span = Self::Span>;

    fn add_macro_def<P, B>(&mut self, name: Self::Span, params: P, body: B) -> Self::MacroDefId
    where
        P: Iterator<Item = Self::Span>,
        B: Iterator<Item = Self::Span>;

    fn mk_buf_context(&mut self, buf_id: I, included_from: Option<Self::Span>) -> Self::BufContext;

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

pub trait BufContext<R = BufRange> {
    type Span: Span;
    fn mk_span(&self, range: R) -> Self::Span;
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
        token: usize,
        expansion: Option<TokenExpansion>,
        context: Rc<MacroExpansionData<SpanData<B, R>>>,
    },
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

impl<B, R> ContextFactory<B, R> for RcContextFactory<B, R>
where
    SpanData<B, R>: Span,
{
    type Span = SpanData<B, R>;
    type MacroDefId = Rc<MacroDef<Self::Span>>;
    type BufContext = RcBufContext<B, R>;
    type MacroExpansionContext = Rc<MacroExpansionData<SpanData<B, R>>>;

    fn add_macro_def<P, C>(&mut self, name: Self::Span, params: P, body: C) -> Self::MacroDefId
    where
        P: Iterator<Item = Self::Span>,
        C: Iterator<Item = Self::Span>,
    {
        Rc::new(MacroDef {
            name,
            params: params.collect(),
            body: body.collect(),
        })
    }

    fn mk_buf_context(&mut self, buf_id: B, included_from: Option<Self::Span>) -> Self::BufContext {
        let context = Rc::new(BufContextData {
            buf_id,
            included_from,
        });
        RcBufContext { context }
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

#[derive(Clone)]
pub struct RcBufContext<B, R> {
    context: Rc<BufContextData<B, R>>,
}

impl<B, R> BufContext<R> for RcBufContext<B, R>
where
    SpanData<B, R>: Span,
{
    type Span = SpanData<B, R>;
    fn mk_span(&self, range: R) -> Self::Span {
        SpanData::Buf {
            range,
            context: self.context.clone(),
        }
    }
}

impl<B, R> MacroExpansionContext for Rc<MacroExpansionData<SpanData<B, R>>> {
    type Span = SpanData<B, R>;

    fn mk_span(&self, token: usize, expansion: Option<TokenExpansion>) -> Self::Span {
        SpanData::Macro {
            token,
            expansion,
            context: Rc::clone(self),
        }
    }
}

impl Span for SpanData<BufId, BufRange> {
    fn extend(&self, other: &Self) -> Self {
        use self::SpanData::*;
        match (self, other) {
            (
                Buf { range, context },
                Buf {
                    range: other_range,
                    context: other_context,
                },
            )
                if Rc::ptr_eq(context, other_context) =>
            {
                Buf {
                    range: cmp::min(range.start, other_range.start)
                        ..cmp::max(range.end, other_range.end),
                    context: (*context).clone(),
                }
            }
            _ => unreachable!(),
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
        let combined = left.extend(&right);
        assert_eq!(
            combined,
            SpanData::Buf {
                range: BufRange::from(0..10),
                context
            }
        )
    }
}
