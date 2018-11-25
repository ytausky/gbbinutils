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

pub trait TokenTracker {
    type Span: Span;
    type BufContext: Clone + BufContext<Span = Self::Span>;
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

#[derive(Clone, Debug, PartialEq)]
pub enum TokenRefData {
    Lexeme {
        range: BufRange,
        context: Rc<BufContextData>,
    },
}

#[derive(Debug, PartialEq)]
pub struct BufContextData {
    pub buf_id: BufId,
    pub included_from: Option<TokenRefData>,
}

pub struct SimpleTokenTracker;

impl TokenTracker for SimpleTokenTracker {
    type Span = TokenRefData;
    type BufContext = SimpleBufTokenRefFactory;
    fn mk_buf_context(
        &mut self,
        buf_id: BufId,
        included_from: Option<Self::Span>,
    ) -> Self::BufContext {
        let context = Rc::new(BufContextData {
            buf_id,
            included_from,
        });
        SimpleBufTokenRefFactory { context }
    }
}

#[derive(Clone)]
pub struct SimpleBufTokenRefFactory {
    context: Rc<BufContextData>,
}

impl BufContext for SimpleBufTokenRefFactory {
    type Span = TokenRefData;
    fn mk_span(&self, range: BufRange) -> Self::Span {
        TokenRefData::Lexeme {
            range,
            context: self.context.clone(),
        }
    }
}

impl Span for TokenRefData {
    fn extend(&self, other: &Self) -> Self {
        use self::TokenRefData::*;
        match (self, other) {
            (
                Lexeme { range, context },
                Lexeme {
                    range: other_range,
                    context: other_context,
                },
            )
                if Rc::ptr_eq(context, other_context) =>
            {
                Lexeme {
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
        let left = TokenRefData::Lexeme {
            range: BufRange::from(0..4),
            context: context.clone(),
        };
        let right = TokenRefData::Lexeme {
            range: BufRange::from(5..10),
            context: context.clone(),
        };
        let combined = left.extend(&right);
        assert_eq!(
            combined,
            TokenRefData::Lexeme {
                range: BufRange::from(0..10),
                context
            }
        )
    }
}
