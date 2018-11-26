use crate::codebase::{BufId, BufRange};
use std::cmp;
use std::fmt::Debug;
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

pub trait ContextFactory {
    type Span: Span;
    type MacroDefId;
    type BufContext: BufContext<Span = Self::Span>;
    type MacroExpansionContext: MacroExpansionContext<Span = Self::Span>;

    fn add_macro_def<P, B>(name: Self::Span, params: P, body: B) -> Self::MacroDefId
    where
        P: Iterator<Item = Self::Span>,
        B: Iterator<Item = Self::Span>;

    fn mk_buf_context(
        &mut self,
        buf_id: BufId,
        included_from: Option<Self::Span>,
    ) -> Self::BufContext;

    fn mk_macro_expansion_context<I, J>(
        &mut self,
        name: Self::Span,
        args: I,
        def: &Self::MacroDefId,
    ) -> Self::MacroExpansionContext
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = Self::Span>;
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
    arg: usize,
    index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SpanData {
    Buf {
        range: BufRange,
        context: Rc<BufContextData>,
    },
    Macro {
        token: usize,
        expansion: Option<TokenExpansion>,
        context: Rc<MacroExpansionData<SpanData>>,
    },
}

#[derive(Debug, PartialEq)]
pub struct BufContextData {
    pub buf_id: BufId,
    pub included_from: Option<SpanData>,
}

#[derive(Debug, PartialEq)]
pub struct MacroExpansionData<S> {
    name: S,
    args: Vec<Vec<S>>,
    def: Rc<MacroDef<S>>,
}

#[derive(Debug, PartialEq)]
pub struct MacroDef<S> {
    name: S,
    params: Vec<S>,
    body: Vec<S>,
}

pub struct RcContextFactory;

impl ContextFactory for RcContextFactory {
    type Span = SpanData;
    type MacroDefId = Rc<MacroDef<Self::Span>>;
    type BufContext = RcBufContext;
    type MacroExpansionContext = RcMacroExpansionContext;

    fn add_macro_def<P, B>(name: Self::Span, params: P, body: B) -> Self::MacroDefId
    where
        P: Iterator<Item = Self::Span>,
        B: Iterator<Item = Self::Span>,
    {
        Rc::new(MacroDef {
            name,
            params: params.collect(),
            body: body.collect(),
        })
    }

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
        RcMacroExpansionContext {
            context: Rc::new(MacroExpansionData {
                name,
                args: args
                    .into_iter()
                    .map(IntoIterator::into_iter)
                    .map(Iterator::collect)
                    .collect(),
                def: Rc::clone(def),
            }),
        }
    }
}

#[derive(Clone)]
pub struct RcBufContext {
    context: Rc<BufContextData>,
}

impl BufContext for RcBufContext {
    type Span = SpanData;
    fn mk_span(&self, range: BufRange) -> Self::Span {
        SpanData::Buf {
            range,
            context: self.context.clone(),
        }
    }
}

pub struct RcMacroExpansionContext {
    context: Rc<MacroExpansionData<SpanData>>,
}

impl MacroExpansionContext for RcMacroExpansionContext {
    type Span = SpanData;

    fn mk_span(&self, token: usize, expansion: Option<TokenExpansion>) -> Self::Span {
        SpanData::Macro {
            token,
            expansion,
            context: Rc::clone(&self.context),
        }
    }
}

impl Span for SpanData {
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
