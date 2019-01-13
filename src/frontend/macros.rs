use super::{Frontend, Ident, SemanticToken, Token};
use crate::diag::span::{MacroContextFactory, MacroExpansionContext, Span};
use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

pub(super) trait MacroTable<I>: Get<Ident<I>> {
    type MacroDefId: Clone;

    fn define<F, S>(
        &mut self,
        name: (Ident<I>, S),
        params: Vec<(Ident<I>, S)>,
        body: Vec<(SemanticToken<I>, S)>,
        factory: &mut F,
    ) where
        F: MacroContextFactory<S, MacroDefId = Self::MacroDefId>,
        Self::Entry: Expand<I, F, S>;
}

pub trait Get<I> {
    type Entry;
    fn get(&self, name: &I) -> Option<&Self::Entry>;
}

pub(super) trait Expand<I, F: MacroContextFactory<S>, S> {
    type Iter: Iterator<Item = (SemanticToken<I>, S)>;

    fn expand(&self, name: S, args: Vec<Vec<(SemanticToken<I>, S)>>, factory: &mut F)
        -> Self::Iter;
}

pub(super) struct MacroExpander<I, D> {
    macro_defs: HashMap<I, MacroTableEntry<D, Rc<MacroDefData<I>>>>,
}

pub struct MacroTableEntry<I, D> {
    id: I,
    def: D,
}

pub struct MacroDefData<I> {
    params: Vec<Ident<I>>,
    body: Vec<SemanticToken<I>>,
}

impl<I: Eq + Hash, D> MacroExpander<I, D> {
    pub fn new() -> MacroExpander<I, D> {
        MacroExpander {
            macro_defs: HashMap::new(),
        }
    }
}

impl<I, D> MacroTable<I> for MacroExpander<I, D>
where
    I: AsRef<str> + Clone + Eq + Hash,
    D: Clone,
{
    type MacroDefId = D;

    fn define<F, S>(
        &mut self,
        name: (Ident<I>, S),
        params: Vec<(Ident<I>, S)>,
        body: Vec<(SemanticToken<I>, S)>,
        factory: &mut F,
    ) where
        F: MacroContextFactory<S, MacroDefId = D>,
    {
        let (param_tokens, param_spans) = split(params);
        let (body_tokens, body_spans) = split(body);
        let id = factory.add_macro_def(name.1, param_spans, body_spans);
        self.macro_defs.insert(
            name.0.name,
            MacroTableEntry {
                id,
                def: Rc::new(MacroDefData {
                    params: param_tokens,
                    body: body_tokens,
                }),
            },
        );
    }
}

impl<I, Id> Get<Ident<I>> for MacroExpander<I, Id>
where
    I: Eq + Hash,
{
    type Entry = MacroTableEntry<Id, Rc<MacroDefData<I>>>;

    fn get(&self, ident: &Ident<I>) -> Option<&Self::Entry> {
        self.macro_defs.get(&ident.name)
    }
}

pub(crate) type MacroEntry<F, D> = MacroTableEntry<
    <D as MacroContextFactory<<D as Span>::Span>>::MacroDefId,
    Rc<MacroDefData<<F as Frontend<D>>::StringRef>>,
>;

impl<I, F, S> Expand<I, F, S> for MacroTableEntry<F::MacroDefId, Rc<MacroDefData<I>>>
where
    I: Clone + Eq,
    F: MacroContextFactory<S>,
{
    type Iter = ExpandedMacro<I, F::MacroExpansionContext>;

    fn expand(
        &self,
        name: S,
        args: Vec<Vec<(SemanticToken<I>, S)>>,
        factory: &mut F,
    ) -> Self::Iter {
        let mut arg_tokens = Vec::new();
        let mut arg_spans = Vec::new();
        for arg in args {
            let (tokens, spans) = split(arg);
            arg_tokens.push(tokens);
            arg_spans.push(spans);
        }
        let context = factory.mk_macro_expansion_context(name, arg_spans, &self.id);
        ExpandedMacro::new(Rc::clone(&self.def), arg_tokens, context)
    }
}

pub(super) struct ExpandedMacro<I, C> {
    def: Rc<MacroDefData<I>>,
    args: Vec<Vec<SemanticToken<I>>>,
    context: C,
    body_index: usize,
    expansion_state: Option<ExpansionState>,
}

#[derive(Debug, PartialEq)]
enum ExpansionState {
    Ident(usize, usize),
    Label(usize),
}

impl<I: PartialEq, C> ExpandedMacro<I, C> {
    fn new(
        def: Rc<MacroDefData<I>>,
        args: Vec<Vec<SemanticToken<I>>>,
        context: C,
    ) -> ExpandedMacro<I, C> {
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
        self.def.params.iter().position(|x| x.name == *name)
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

    fn expand_token(&self, token: &SemanticToken<I>) -> Option<ExpansionState> {
        match token {
            Token::Ident(ident) => self
                .param_position(&ident.name)
                .map(|position| ExpansionState::Ident(position, 0)),
            Token::Label(label) => self.param_position(&label.name).map(ExpansionState::Label),
            _ => None,
        }
    }
}

impl<I, C> Iterator for ExpandedMacro<I, C>
where
    I: Clone + Eq,
    C: MacroExpansionContext,
{
    type Item = (SemanticToken<I>, C::Span);

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
            let item = (token, self.context.mk_span(self.body_index, expansion));
            self.advance();
            Some(item)
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
    use crate::span::*;

    #[test]
    fn expand_macro_with_one_token() {
        let buf = Rc::new(BufContextData {
            buf_id: (),
            included_from: None,
        });
        let mk_span = |n| SpanData::Buf {
            range: n,
            context: Rc::clone(&buf),
        };
        let mut table = MacroExpander::<&'static str, _>::new();
        let def_name = ("my_macro".into(), mk_span(0));
        let body = (Token::Ident("a".into()), mk_span(1));
        let def_id = Rc::new(crate::span::MacroDef {
            name: mk_span(0),
            params: vec![],
            body: vec![mk_span(1)],
        });
        let mut factory = RcContextFactory::new();
        table.define(def_name, vec![], vec![body.clone()], &mut factory);
        let invocation_name = ("my_macro".into(), mk_span(2));
        let data = Rc::new(MacroExpansionData {
            name: invocation_name.1.clone(),
            args: vec![],
            def: def_id,
        });
        let expanded: Vec<_> = table
            .get(&invocation_name.0)
            .unwrap()
            .expand(invocation_name.1, vec![], &mut factory)
            .collect();
        let macro_expansion_position = MacroExpansionPosition {
            token: 0,
            expansion: None,
        };
        assert_eq!(
            expanded,
            [(
                body.0,
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
        let mk_span = |n| SpanData::Buf {
            range: n,
            context: Rc::clone(&buf),
        };
        let mut table = MacroExpander::<&'static str, _>::new();
        let def_name = ("my_macro".into(), mk_span(0));
        let body = vec![
            Token::Ident("a".into()),
            Token::Ident("x".into()),
            Token::Ident("b".into()),
        ]
        .into_iter()
        .zip((2..=4).map(mk_span));
        let def_id = Rc::new(crate::span::MacroDef {
            name: def_name.1.clone(),
            params: vec![mk_span(1)],
            body: (2..=4).map(mk_span).collect(),
        });
        let factory = &mut RcContextFactory::new();
        table.define(
            def_name,
            vec![("x".into(), mk_span(1))],
            body.collect(),
            factory,
        );
        let data = Rc::new(MacroExpansionData {
            name: SpanData::Buf {
                range: 7,
                context: buf.clone(),
            },
            args: vec![vec![
                SpanData::Buf {
                    range: 8,
                    context: buf.clone(),
                },
                SpanData::Buf {
                    range: 9,
                    context: buf.clone(),
                },
            ]],
            def: def_id,
        });
        let invocation_name = ("my_macro".into(), mk_span(7));
        let expanded: Vec<_> = table
            .get(&invocation_name.0)
            .unwrap()
            .expand(
                invocation_name.1,
                vec![vec![Token::Ident("y".into()), Token::Ident("z".into())]
                    .into_iter()
                    .zip((8..=9).map(mk_span))
                    .collect()],
                factory,
            )
            .collect();
        let mk_span_data = |token, expansion| {
            let position = MacroExpansionPosition { token, expansion };
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
                    mk_span_data(1, Some(TokenExpansion { arg: 0, index: 0 })),
                ),
                (
                    Token::Ident("z".into()),
                    mk_span_data(1, Some(TokenExpansion { arg: 0, index: 1 })),
                ),
                (Token::Ident("b".into()), mk_span_data(2, None)),
            ]
        )
    }
}
