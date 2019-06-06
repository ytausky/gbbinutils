use crate::codebase::{BufId, BufRange, TextCache};

use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::{Deref, Range, RangeInclusive};
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
    type MacroDefHandle: Clone;
    type MacroCallCtx: MacroCallCtx<Span = S>;

    fn add_macro_def<P, B>(&mut self, name: S, params: P, body: B) -> Self::MacroDefHandle
    where
        P: IntoIterator<Item = S>,
        B: IntoIterator<Item = S>;

    fn mk_macro_call_ctx<A, J>(
        &mut self,
        name: S,
        args: A,
        def: &Self::MacroDefHandle,
    ) -> Self::MacroCallCtx
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

pub trait MacroCallCtx: SpanSource {
    fn mk_span(&self, position: MacroCallPos) -> Self::Span;
}

#[derive(Clone, Debug, PartialEq)]
pub struct MacroCallPos {
    pub token: usize,
    pub expansion: Option<ArgExpansionPos>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArgExpansionPos {
    pub arg: usize,
    pub token: usize,
}

pub type RcSpan<B, R> = ModularSpan<BufSpan<B, R>, MacroSpan<RcMacroCall<BufSpan<B, R>>>>;

#[derive(Clone, Debug, PartialEq)]
pub enum ModularSpan<B, M> {
    Buf(B),
    Macro(M),
}

#[derive(Clone, Debug, PartialEq)]
pub struct BufSpan<B, R> {
    pub range: R,
    pub context: Rc<BufContextData<B, R>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MacroSpan<C> {
    pub range: RangeInclusive<MacroCallPos>,
    pub context: C,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StrippedBufSpan<B, R> {
    pub buf_id: B,
    pub range: R,
}

impl<B: Clone, T: Clone> RcSpan<B, Range<T>> {
    pub fn to_stripped(&self) -> StrippedBufSpan<B, Range<T>> {
        match self {
            ModularSpan::Buf(BufSpan { range, context }) => StrippedBufSpan {
                buf_id: context.buf_id.clone(),
                range: range.clone(),
            },
            ModularSpan::Macro(MacroSpan { range, context }) => {
                let start = &context.def.body[range.start().token];
                let end = &context.def.body[range.end().token];
                let (buf_id, range) = match (start, end) {
                    (
                        ModularSpan::Buf(BufSpan {
                            ref range,
                            ref context,
                        }),
                        ModularSpan::Buf(BufSpan {
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
    pub fn snippet(&self, stripped: &StrippedBufSpan<BufId, BufRange>) -> &str {
        &self.buf(stripped.buf_id).as_str()[stripped.range.clone()]
    }
}

impl PartialOrd for MacroCallPos {
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
    left: &RangeInclusive<MacroCallPos>,
    right: &RangeInclusive<MacroCallPos>,
) -> RangeInclusive<MacroCallPos> {
    assert!(left.start() <= right.end());
    left.start().clone()..=right.end().clone()
}

#[derive(Debug, PartialEq)]
pub struct BufContextData<B, R> {
    pub buf_id: B,
    pub included_from: Option<RcSpan<B, R>>,
}

#[derive(Debug, PartialEq)]
pub struct RcMacroCall<B>(Rc<RcMacroCallInner<B>>);

type RcMacroCallInner<B> = ModularMacroCall<
    Rc<MacroDefSpans<ModularSpan<B, MacroSpan<RcMacroCall<B>>>>>,
    ModularSpan<B, MacroSpan<RcMacroCall<B>>>,
>;

impl<B> RcMacroCall<B> {
    pub fn new(inner: RcMacroCallInner<B>) -> Self {
        Self(Rc::new(inner))
    }
}

impl<B> Clone for RcMacroCall<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<B> Deref for RcMacroCall<B> {
    type Target = RcMacroCallInner<B>;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
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

pub struct RcContextFactory<B = BufId, R = BufRange>(PhantomData<(B, R)>);

impl<B, R> RcContextFactory<B, R> {
    pub fn new() -> Self {
        RcContextFactory(PhantomData)
    }
}

impl<B, R> SpanSource for RcContextFactory<B, R>
where
    RcSpan<B, R>: Clone,
{
    type Span = RcSpan<B, R>;
}

impl<B, R> MacroContextFactory<RcSpan<B, R>> for RcContextFactory<B, R>
where
    RcSpan<B, R>: Clone,
{
    type MacroDefHandle = Rc<MacroDefSpans<RcSpan<B, R>>>;
    type MacroCallCtx = RcMacroCall<BufSpan<B, R>>;

    fn add_macro_def<P, C>(
        &mut self,
        name: RcSpan<B, R>,
        params: P,
        body: C,
    ) -> Self::MacroDefHandle
    where
        P: IntoIterator<Item = RcSpan<B, R>>,
        C: IntoIterator<Item = RcSpan<B, R>>,
    {
        Rc::new(MacroDefSpans {
            name,
            params: params.into_iter().collect(),
            body: body.into_iter().collect(),
        })
    }

    fn mk_macro_call_ctx<I, J>(
        &mut self,
        name: RcSpan<B, R>,
        args: I,
        def: &Self::MacroDefHandle,
    ) -> Self::MacroCallCtx
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = RcSpan<B, R>>,
    {
        RcMacroCall::new(ModularMacroCall {
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

impl MergeSpans<RcSpan<BufId, BufRange>> for RcContextFactory<BufId, BufRange> {
    fn merge_spans(
        &mut self,
        left: &RcSpan<BufId, BufRange>,
        right: &RcSpan<BufId, BufRange>,
    ) -> RcSpan<BufId, BufRange> {
        use self::ModularSpan::*;
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
                Macro(MacroSpan { range, context }),
                Macro(MacroSpan {
                    range: other_range,
                    context: other_context,
                }),
            ) if Rc::ptr_eq(&context.0, &other_context.0) => Macro(MacroSpan {
                range: merge_macro_expansion_ranges(range, other_range),
                context: context.clone(),
            }),
            _ => unreachable!(),
        }
    }
}

impl StripSpan<RcSpan<BufId, BufRange>> for RcContextFactory<BufId, BufRange> {
    type Stripped = StrippedBufSpan<BufId, BufRange>;

    fn strip_span(&mut self, span: &RcSpan<BufId, BufRange>) -> Self::Stripped {
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
    type Span = RcSpan<BufId, BufRange>;
}

impl BufContext for RcBufContext<BufId, BufRange> {
    fn mk_span(&self, range: BufRange) -> Self::Span {
        ModularSpan::Buf(BufSpan {
            range,
            context: self.context.clone(),
        })
    }
}

impl<B, R> SpanSource for RcMacroCall<BufSpan<B, R>>
where
    RcSpan<B, R>: Clone,
{
    type Span = RcSpan<B, R>;
}

impl<B, R> MacroCallCtx for RcMacroCall<BufSpan<B, R>>
where
    RcSpan<B, R>: Clone,
{
    fn mk_span(&self, position: MacroCallPos) -> Self::Span {
        ModularSpan::Macro(MacroSpan {
            range: position.clone()..=position,
            context: self.clone(),
        })
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
        let left = ModularSpan::Buf(BufSpan {
            range: 0..4,
            context: context.clone(),
        });
        let right = ModularSpan::Buf(BufSpan {
            range: 5..10,
            context: context.clone(),
        });
        let combined = RcContextFactory::new().merge_spans(&left, &right);
        assert_eq!(
            combined,
            ModularSpan::Buf(BufSpan {
                range: 0..10,
                context
            })
        )
    }

    #[test]
    fn merge_simple_macro_expansion_positions() {
        let start_pos = MacroCallPos {
            token: 2,
            expansion: None,
        };
        let start = start_pos.clone()..=start_pos.clone();
        let end_pos = MacroCallPos {
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
        let span = ModularSpan::Buf(BufSpan {
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
        let position = MacroCallPos {
            token: 0,
            expansion: None,
        };
        let macro_base = 0;
        let expansion = ModularMacroCall {
            name: ModularSpan::Buf(BufSpan {
                range: 40..50,
                context: Rc::clone(&buf_context),
            }),
            args: vec![],
            def: mk_macro_def(&buf_context, macro_base),
        };
        let span = ModularSpan::Macro(MacroSpan {
            range: position.clone()..=position,
            context: RcMacroCall(Rc::new(expansion)),
        });
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
    ) -> Rc<MacroDefSpans<RcSpan<B, BufRange>>> {
        Rc::new(MacroDefSpans {
            name: ModularSpan::Buf(BufSpan {
                range: macro_name_range(base),
                context: Rc::clone(buf_context),
            }),
            params: vec![],
            body: vec![ModularSpan::Buf(BufSpan {
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
