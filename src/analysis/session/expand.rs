use crate::analysis::resolve::{Name, NameTable};
use crate::analysis::Token;
use crate::diag::span::*;

use std::rc::Rc;

pub(super) trait Expand<T, H, F: MacroContextFactory<H, S>, S: Clone> {
    type Iter: Iterator<Item = (T, S)>;

    fn expand(&self, name: S, args: MacroArgs<T, S>, factory: &mut F) -> Self::Iter;
}

pub(super) type MacroArgs<T, S> = (Vec<Vec<T>>, Vec<Vec<S>>);

pub(super) trait DefineMacro<I, T, H: Clone> {
    fn define_macro<D, S>(
        &mut self,
        name: (I, S),
        params: (Vec<I>, Vec<S>),
        body: (Vec<T>, Vec<S>),
        diagnostics: &mut D,
    ) where
        D: AddMacroDef<S, MacroDefHandle = H> + MacroContextFactory<H, S>,
        S: Clone;
}

impl<I, L, C, N, H> DefineMacro<I, Token<I, L, C>, H> for N
where
    N: NameTable<I, MacroEntry = MacroDef<I, Token<I, L, C>, H>>,
    H: Clone,
{
    fn define_macro<D, S>(
        &mut self,
        name: (I, S),
        params: (Vec<I>, Vec<S>),
        body: (Vec<Token<I, L, C>>, Vec<S>),
        diagnostics: &mut D,
    ) where
        D: AddMacroDef<S, MacroDefHandle = H> + MacroContextFactory<H, S>,
        S: Clone,
    {
        let context = diagnostics.add_macro_def(name.1, params.1, body.1);
        self.insert(
            name.0,
            Name::Macro(MacroDef {
                tokens: Rc::new(MacroDefTokens {
                    params: params.0,
                    body: body.0,
                }),
                spans: context,
            }),
        )
    }
}

pub(in crate::analysis) struct MacroDef<I, T, S> {
    pub tokens: Rc<MacroDefTokens<I, T>>,
    pub spans: S,
}

pub(in crate::analysis) struct MacroDefTokens<I, T> {
    pub params: Vec<I>,
    pub body: Vec<T>,
}

impl<I, L, C, H, F, S> Expand<Token<I, L, C>, H, F, S> for MacroDef<I, Token<I, L, C>, H>
where
    I: Clone + Eq,
    F: MacroContextFactory<H, S>,
    S: Clone,
    Token<I, L, C>: Clone,
{
    type Iter = MacroExpansionIter<I, Token<I, L, C>, F::MacroCallCtx>;

    fn expand(
        &self,
        name: S,
        (args, arg_spans): MacroArgs<Token<I, L, C>, S>,
        factory: &mut F,
    ) -> Self::Iter {
        let context = factory.mk_macro_call_ctx(name, arg_spans, &self.spans);
        MacroExpansionIter::new(self.tokens.clone(), args, context)
    }
}

pub(super) struct MacroExpansionIter<I, T, C> {
    expansion: MacroExpansion<I, T, C>,
    pos: MacroCallPos,
}

struct MacroExpansion<I, T, C> {
    def: Rc<MacroDefTokens<I, T>>,
    args: Vec<Vec<T>>,
    context: C,
}

impl<I: PartialEq, L, C, F> MacroExpansion<I, Token<I, L, C>, F> {
    fn expand_token(&self, token: usize) -> Option<usize> {
        let name = match &self.def.body[token] {
            Token::Ident(name) | Token::Label(name) => Some(name),
            _ => None,
        };
        name.and_then(|name| self.param_position(name))
    }

    fn param_position(&self, name: &I) -> Option<usize> {
        self.def.params.iter().position(|param| *param == *name)
    }
}

impl<I, L, C, F> MacroExpansionIter<I, Token<I, L, C>, F>
where
    I: PartialEq,
{
    fn new(
        def: Rc<MacroDefTokens<I, Token<I, L, C>>>,
        args: Vec<Vec<Token<I, L, C>>>,
        context: F,
    ) -> Self {
        let mut expanded_macro = MacroExpansionIter {
            expansion: MacroExpansion { def, args, context },
            pos: MacroCallPos {
                token: 0,
                expansion: None,
            },
        };
        expanded_macro.try_expand();
        expanded_macro
    }

    fn advance(&mut self) {
        self.pos.expansion = match self.pos.expansion {
            Some(ParamExpansionPos { param, arg_token })
                if arg_token + 1 < self.expansion.args[param].len() =>
            {
                Some(ParamExpansionPos {
                    param,
                    arg_token: arg_token + 1,
                })
            }
            _ => None,
        };
        if self.pos.expansion.is_none() {
            self.pos.token += 1;
            self.try_expand()
        }
    }

    fn try_expand(&mut self) {
        assert_eq!(self.pos.expansion, None);
        if self.pos.token < self.expansion.def.body.len() {
            self.pos.expansion =
                self.expansion
                    .expand_token(self.pos.token)
                    .map(|param| ParamExpansionPos {
                        param,
                        arg_token: 0,
                    })
        }
    }
}

impl<I, L, C, F> Iterator for MacroExpansionIter<I, Token<I, L, C>, F>
where
    I: Clone + Eq,
    F: MacroCallCtx,
    Token<I, L, C>: Clone,
{
    type Item = (Token<I, L, C>, F::Span);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos.token < self.expansion.def.body.len() {
            let token = match self.pos.expansion {
                Some(pos) => match (
                    &self.expansion.def.body[self.pos.token],
                    &self.expansion.args[pos.param][pos.arg_token],
                ) {
                    (Token::Label(_), Token::Ident(ident)) if pos.arg_token == 0 => {
                        Token::Label(ident.clone())
                    }
                    (_, arg_token) => arg_token.clone(),
                },
                None => self.expansion.def.body[self.pos.token].clone(),
            };
            let span = self.expansion.context.mk_span(self.pos.clone());
            self.advance();
            Some((token, span))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_macro_with_one_token() {
        let body = Token::<_, (), ()>::Ident("a");
        let entry = MacroDef {
            tokens: Rc::new(MacroDefTokens {
                params: vec![],
                body: vec![body.clone()],
            }),
            spans: (),
        };
        let name = ModularSpan::Buf(());
        let expanded: Vec<_> = entry
            .expand(name.clone(), (vec![], vec![]), &mut Factory)
            .collect();
        let data = MacroCall(Rc::new(ModularMacroCall {
            name,
            args: vec![],
            def: (),
        }));
        let macro_expansion_position = MacroCallPos {
            token: 0,
            expansion: None,
        };
        assert_eq!(
            expanded,
            [(
                body,
                ModularSpan::Macro(MacroSpan {
                    range: macro_expansion_position.clone()..=macro_expansion_position,
                    context: data,
                })
            )]
        )
    }

    #[test]
    fn expand_label_using_two_idents() {
        let label = Token::<_, (), ()>::Label("label");
        let def = MacroDef {
            tokens: Rc::new(MacroDefTokens {
                params: vec!["label"],
                body: vec![label],
            }),
            spans: (),
        };
        let name = ModularSpan::Buf(());
        let arg = vec![Token::Ident("tok1"), Token::Ident("tok2")];
        let expanded: Vec<_> = def
            .expand(
                name.clone(),
                (
                    vec![arg],
                    vec![vec![ModularSpan::Buf(()), ModularSpan::Buf(())]],
                ),
                &mut Factory,
            )
            .collect();
        let context = MacroCall(Rc::new(ModularMacroCall {
            name,
            args: vec![vec![ModularSpan::Buf(()), ModularSpan::Buf(())]],
            def: (),
        }));
        let tok1_pos = MacroCallPos {
            token: 0,
            expansion: Some(ParamExpansionPos {
                param: 0,
                arg_token: 0,
            }),
        };
        let tok2_pos = MacroCallPos {
            token: 0,
            expansion: Some(ParamExpansionPos {
                param: 0,
                arg_token: 1,
            }),
        };
        assert_eq!(
            expanded,
            [
                (
                    Token::Label("tok1"),
                    ModularSpan::Macro(MacroSpan {
                        range: tok1_pos.clone()..=tok1_pos,
                        context: context.clone()
                    })
                ),
                (
                    Token::Ident("tok2"),
                    ModularSpan::Macro(MacroSpan {
                        range: tok2_pos.clone()..=tok2_pos,
                        context,
                    })
                )
            ]
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
            ModularSpan::Buf(BufSpan {
                range: n,
                context: Rc::clone(&buf),
            })
        };
        let body: Vec<Token<_, (), ()>> =
            vec![Token::Ident("a"), Token::Ident("x"), Token::Ident("b")];
        let def_id = Rc::new(MacroDefSpans {
            name: mk_span(0),
            params: vec![mk_span(1)],
            body: (2..=4).map(mk_span).collect(),
        });
        let factory = &mut RcContextFactory::new();
        let entry = MacroDef {
            tokens: Rc::new(MacroDefTokens {
                params: vec!["x"],
                body,
            }),
            spans: Rc::clone(&def_id),
        };
        let data = RcMacroCall::new(ModularMacroCall {
            name: ModularSpan::Buf(BufSpan {
                range: 7,
                context: buf.clone(),
            }),
            args: vec![vec![
                ModularSpan::Buf(BufSpan {
                    range: 8,
                    context: buf.clone(),
                }),
                ModularSpan::Buf(BufSpan {
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
                (
                    vec![vec![Token::Ident("y"), Token::Ident("z")]],
                    vec![(8..=9).map(mk_span).collect()],
                ),
                factory,
            )
            .collect();
        let mk_span_data = |token, expansion| {
            let position = MacroCallPos { token, expansion };
            ModularSpan::Macro(MacroSpan {
                range: position.clone()..=position,
                context: data.clone(),
            })
        };
        assert_eq!(
            expanded,
            [
                (Token::Ident("a"), mk_span_data(0, None)),
                (
                    Token::Ident("y"),
                    mk_span_data(
                        1,
                        Some(ParamExpansionPos {
                            param: 0,
                            arg_token: 0
                        })
                    ),
                ),
                (
                    Token::Ident("z"),
                    mk_span_data(
                        1,
                        Some(ParamExpansionPos {
                            param: 0,
                            arg_token: 1
                        })
                    ),
                ),
                (Token::Ident("b"), mk_span_data(2, None)),
            ]
        )
    }

    #[derive(Clone, Debug, PartialEq)]
    struct MacroCall(Rc<ModularMacroCall<(), Span>>);

    type Span = ModularSpan<(), MacroSpan<MacroCall>>;

    struct Factory;

    impl MacroContextFactory<(), Span> for Factory {
        type MacroCallCtx = MacroCall;

        fn mk_macro_call_ctx<A, J>(&mut self, name: Span, args: A, _: &()) -> Self::MacroCallCtx
        where
            A: IntoIterator<Item = J>,
            J: IntoIterator<Item = Span>,
        {
            MacroCall(Rc::new(ModularMacroCall {
                name,
                args: args
                    .into_iter()
                    .map(IntoIterator::into_iter)
                    .map(Iterator::collect)
                    .collect(),
                def: (),
            }))
        }
    }

    impl SpanSource for MacroCall {
        type Span = Span;
    }

    impl MacroCallCtx for MacroCall {
        fn mk_span(&self, position: MacroCallPos) -> Self::Span {
            ModularSpan::Macro(MacroSpan {
                range: position.clone()..=position,
                context: self.clone(),
            })
        }
    }
}
