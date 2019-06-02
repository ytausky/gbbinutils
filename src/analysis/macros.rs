use super::resolve::{Name, NameTable};
use super::Token;

use crate::diag::span::*;

use std::ops::Deref;
use std::rc::Rc;

pub(super) trait Expand<T, F: MacroContextFactory<S>, S: Clone> {
    type Iter: Iterator<Item = (T, S)>;

    fn expand(&self, name: S, args: MacroArgs<T, S>, factory: &mut F) -> Self::Iter;
}

type MacroArgs<T, S> = Vec<Vec<(T, S)>>;

pub(super) trait DefineMacro<I, T, K: Clone> {
    fn define_macro<D, S>(
        &mut self,
        name: (I, S),
        params: (Vec<I>, Vec<S>),
        body: (Vec<T>, Vec<S>),
        diagnostics: &mut D,
    ) where
        D: MacroContextFactory<S, MacroDefId = K>,
        S: Clone;
}

impl<I, L, C, N, K> DefineMacro<I, Token<I, L, C>, K> for N
where
    N: NameTable<I, MacroEntry = MacroTableEntry<K, Rc<MacroDefData<I, Token<I, L, C>>>>>,
    K: Clone,
{
    fn define_macro<D, S>(
        &mut self,
        name: (I, S),
        params: (Vec<I>, Vec<S>),
        body: (Vec<Token<I, L, C>>, Vec<S>),
        diagnostics: &mut D,
    ) where
        D: MacroContextFactory<S, MacroDefId = K>,
        S: Clone,
    {
        let context = diagnostics.add_macro_def(name.1, params.1, body.1);
        self.insert(
            name.0,
            Name::Macro(MacroTableEntry {
                id: context,
                def: Rc::new(MacroDefData {
                    params: params.0,
                    body: body.0,
                }),
            }),
        )
    }
}

pub struct MacroTableEntry<I, D> {
    pub id: I,
    pub def: D,
}

pub(crate) struct MacroDefData<I, T> {
    pub params: Vec<I>,
    pub body: Vec<T>,
}

impl<I, L, C, M, F, S> Expand<Token<I, L, C>, F, S> for MacroTableEntry<F::MacroDefId, M>
where
    I: Clone + Eq,
    M: Clone + Deref<Target = MacroDefData<I, Token<I, L, C>>>,
    F: MacroContextFactory<S>,
    S: Clone,
    Token<I, L, C>: Clone,
{
    type Iter = ExpandedMacro<M, Token<I, L, C>, F::MacroExpansionContext>;

    fn expand(&self, name: S, args: MacroArgs<Token<I, L, C>, S>, factory: &mut F) -> Self::Iter {
        let mut arg_tokens = Vec::new();
        let mut arg_spans = Vec::new();
        for arg in args {
            let (tokens, spans) = split(arg);
            arg_tokens.push(tokens);
            arg_spans.push(spans);
        }
        let context = factory.mk_macro_expansion_context(name, arg_spans, &self.id);
        ExpandedMacro::new(self.def.clone(), arg_tokens, context)
    }
}

pub(super) struct ExpandedMacro<M, T, C> {
    def: M,
    args: Vec<Vec<T>>,
    context: C,
    body_index: usize,
    expansion_state: Option<ExpansionState>,
}

#[derive(Debug, PartialEq)]
enum ExpansionState {
    Ident(usize, usize),
    Label(usize),
}

impl<M, I, L, C, F> ExpandedMacro<M, Token<I, L, C>, F>
where
    M: Deref<Target = MacroDefData<I, Token<I, L, C>>>,
    I: PartialEq,
{
    fn new(def: M, args: Vec<Vec<Token<I, L, C>>>, context: F) -> Self {
        let mut expanded_macro = ExpandedMacro {
            def,
            args,
            context,
            body_index: 0,
            expansion_state: None,
        };
        expanded_macro.try_expand();
        expanded_macro
    }

    fn param_position(&self, name: &I) -> Option<usize> {
        self.def.params.iter().position(|param| *param == *name)
    }

    fn advance(&mut self) {
        self.expansion_state = match self.expansion_state {
            Some(ExpansionState::Ident(position, index))
                if index + 1 < self.args[position].len() =>
            {
                Some(ExpansionState::Ident(position, index + 1))
            }
            _ => None,
        };
        if self.expansion_state.is_none() {
            self.body_index += 1;
            self.try_expand()
        }
    }

    fn try_expand(&mut self) {
        assert_eq!(self.expansion_state, None);
        if self.body_index < self.def.body.len() {
            self.expansion_state = self.expand_token(&self.def.body[self.body_index]);
        }
    }

    fn expand_token(&self, token: &Token<I, L, C>) -> Option<ExpansionState> {
        match token {
            Token::Ident(ident) => self
                .param_position(&ident)
                .map(|position| ExpansionState::Ident(position, 0)),
            Token::Label(label) => self.param_position(&label).map(ExpansionState::Label),
            _ => None,
        }
    }
}

impl<M, I, L, C, F> Iterator for ExpandedMacro<M, Token<I, L, C>, F>
where
    M: Deref<Target = MacroDefData<I, Token<I, L, C>>>,
    I: Clone + Eq,
    F: MacroExpansionContext,
    Token<I, L, C>: Clone,
{
    type Item = (Token<I, L, C>, F::Span);

    fn next(&mut self) -> Option<Self::Item> {
        if self.body_index < self.def.body.len() {
            let (token, expansion) = match self.expansion_state {
                Some(ExpansionState::Ident(position, index)) => {
                    (self.args[position][index].clone(), None)
                }
                Some(ExpansionState::Label(position)) => match self.args[position][0] {
                    Token::Ident(ref ident) => (Token::Label(ident.clone()), None),
                    _ => unimplemented!(),
                },
                None => (self.def.body[self.body_index].clone(), None),
            };
            let span = self.context.mk_span(MacroExpansionPos {
                token: self.body_index,
                expansion,
            });
            self.advance();
            Some((token, span))
        } else {
            None
        }
    }
}

fn split<I: IntoIterator<Item = (L, R)>, L, R>(iter: I) -> (Vec<L>, Vec<R>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for (l, r) in iter {
        left.push(l);
        right.push(r);
    }
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_macro_with_one_token() {
        let buf = Rc::new(BufContextData {
            buf_id: (),
            included_from: None,
        });
        let mk_span = |n| {
            SpanData::Buf(BufSpan {
                range: n,
                context: Rc::clone(&buf),
            })
        };
        let body = Token::<_, (), ()>::Ident("a");
        let def_id = Rc::new(crate::span::MacroDef {
            name: mk_span(0),
            params: vec![],
            body: vec![mk_span(1)],
        });
        let mut factory = RcContextFactory::new();
        let entry = MacroTableEntry {
            id: Rc::clone(&def_id),
            def: Rc::new(MacroDefData {
                params: vec![],
                body: vec![body.clone()],
            }),
        };
        let call_name = ("my_macro", mk_span(2));
        let data = Rc::new(MacroExpansionData {
            name: call_name.1.clone(),
            args: vec![],
            def: def_id,
        });
        let expanded: Vec<_> = entry.expand(call_name.1, vec![], &mut factory).collect();
        let macro_expansion_position = MacroExpansionPos {
            token: 0,
            expansion: None,
        };
        assert_eq!(
            expanded,
            [(
                body,
                SpanData::Macro {
                    range: macro_expansion_position.clone()..=macro_expansion_position,
                    context: data,
                }
            )]
        )
    }

    #[ignore]
    #[test]
    fn expand_macro() {
        let buf = Rc::new(BufContextData {
            buf_id: (),
            included_from: None,
        });
        let mk_span = |n| {
            SpanData::Buf(BufSpan {
                range: n,
                context: Rc::clone(&buf),
            })
        };
        let body: Vec<Token<_, (), ()>> =
            vec![Token::Ident("a"), Token::Ident("x"), Token::Ident("b")];
        let def_id = Rc::new(crate::span::MacroDef {
            name: mk_span(0),
            params: vec![mk_span(1)],
            body: (2..=4).map(mk_span).collect(),
        });
        let factory = &mut RcContextFactory::new();
        let entry = MacroTableEntry {
            id: Rc::clone(&def_id),
            def: Rc::new(MacroDefData {
                params: vec!["x".into()],
                body,
            }),
        };
        let data = Rc::new(MacroExpansionData {
            name: SpanData::Buf(BufSpan {
                range: 7,
                context: buf.clone(),
            }),
            args: vec![vec![
                SpanData::Buf(BufSpan {
                    range: 8,
                    context: buf.clone(),
                }),
                SpanData::Buf(BufSpan {
                    range: 9,
                    context: buf.clone(),
                }),
            ]],
            def: def_id,
        });
        let call_name = ("my_macro", mk_span(7));
        let expanded: Vec<_> = entry
            .expand(
                call_name.1,
                vec![vec![Token::Ident("y".into()), Token::Ident("z".into())]
                    .into_iter()
                    .zip((8..=9).map(mk_span))
                    .collect()],
                factory,
            )
            .collect();
        let mk_span_data = |token, expansion| {
            let position = MacroExpansionPos { token, expansion };
            SpanData::Macro {
                range: position.clone()..=position,
                context: data.clone(),
            }
        };
        assert_eq!(
            expanded,
            [
                (Token::Ident("a".into()), mk_span_data(0, None)),
                (
                    Token::Ident("y".into()),
                    mk_span_data(1, Some(ArgExpansionPos { arg: 0, token: 0 })),
                ),
                (
                    Token::Ident("z".into()),
                    mk_span_data(1, Some(ArgExpansionPos { arg: 0, token: 1 })),
                ),
                (Token::Ident("b".into()), mk_span_data(2, None)),
            ]
        )
    }
}
