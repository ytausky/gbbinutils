use crate::analyze::semantics::session::reentrancy::SourceComponents;
use crate::analyze::Token;
use crate::diag::span::*;
use crate::CompositeSession;

use std::ops::DerefMut;
use std::rc::Rc;

pub trait MacroSource {
    type MacroId: Clone;
}

pub(super) trait MacroTable<I, L, S: Clone>: MacroSource {
    type Iter: Iterator<Item = (Token<I, L>, S)>;

    fn define_macro(
        &mut self,
        name_span: S,
        params: (Vec<I>, Vec<S>),
        body: (Vec<Token<I, L>>, Vec<S>),
    ) -> Self::MacroId;

    fn expand_macro(
        &mut self,
        name: (Self::MacroId, S),
        args: MacroArgs<Token<I, L>, S>,
    ) -> Self::Iter;
}

pub(super) type VecMacroTable<I, L, H> = Vec<MacroDef<I, Token<I, L>, H>>;

pub(super) type MacroArgs<T, S> = (Vec<Vec<T>>, Vec<Vec<S>>);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacroId(usize);

impl<'a, C, P, J, D, I, L> MacroSource
    for SourceComponents<
        C,
        P,
        &'a mut VecMacroTable<
            I,
            L,
            <D::Target as AddMacroDef<<D::Target as SpanSource>::Span>>::MacroDefHandle,
        >,
        J,
        D,
    >
where
    D: DerefMut,
    D::Target: SpanSource
        + AddMacroDef<<D::Target as SpanSource>::Span>
        + MacroContextFactory<
            <D::Target as AddMacroDef<<D::Target as SpanSource>::Span>>::MacroDefHandle,
            <D::Target as SpanSource>::Span,
        >,
{
    type MacroId = MacroId;
}

impl<'a, C, P, J, D, N, B, I, L> MacroTable<I, L, <D::Target as SpanSource>::Span>
    for CompositeSession<
        SourceComponents<
            C,
            P,
            &'a mut VecMacroTable<
                I,
                L,
                <D::Target as AddMacroDef<<D::Target as SpanSource>::Span>>::MacroDefHandle,
            >,
            J,
            D,
        >,
        N,
        B,
    >
where
    D: DerefMut,
    D::Target: SpanSource
        + AddMacroDef<<D::Target as SpanSource>::Span>
        + MacroContextFactory<
            <D::Target as AddMacroDef<<D::Target as SpanSource>::Span>>::MacroDefHandle,
            <D::Target as SpanSource>::Span,
        >,
    I: Clone + PartialEq,
    L: Clone,
{
    type Iter = MacroExpansionIter<
        I,
        Token<I, L>,
        <D::Target as MacroContextFactory<
            <D::Target as AddMacroDef<<D::Target as SpanSource>::Span>>::MacroDefHandle,
            <D::Target as SpanSource>::Span,
        >>::MacroCallCtx,
    >;

    fn define_macro(
        &mut self,
        name_span: <D::Target as SpanSource>::Span,
        params: (Vec<I>, Vec<<D::Target as SpanSource>::Span>),
        body: (Vec<Token<I, L>>, Vec<<D::Target as SpanSource>::Span>),
    ) -> Self::MacroId {
        let context = self
            .reentrancy
            .diagnostics
            .add_macro_def(name_span, params.1, body.1);
        let id = MacroId(self.reentrancy.macros.len());
        self.reentrancy.macros.push(MacroDef {
            tokens: Rc::new(MacroDefTokens {
                params: params.0,
                body: body.0,
            }),
            spans: context,
        });
        id
    }

    fn expand_macro(
        &mut self,
        (MacroId(id), name_span): (Self::MacroId, <D::Target as SpanSource>::Span),
        (args, arg_spans): MacroArgs<Token<I, L>, <D::Target as SpanSource>::Span>,
    ) -> Self::Iter {
        let def = &self.reentrancy.macros[id];
        let context = self
            .reentrancy
            .diagnostics
            .mk_macro_call_ctx(name_span, arg_spans, &def.spans);
        MacroExpansionIter::new(def.tokens.clone(), args, context)
    }
}

pub(in crate::analyze) struct MacroDef<I, T, S> {
    tokens: Rc<MacroDefTokens<I, T>>,
    spans: S,
}

struct MacroDefTokens<I, T> {
    params: Vec<I>,
    body: Vec<T>,
}

pub(super) struct MacroExpansionIter<I, T, C> {
    expansion: MacroExpansion<I, T, C>,
    pos: Option<MacroExpansionPos>,
}

struct MacroExpansion<I, T, C> {
    def: Rc<MacroDefTokens<I, T>>,
    args: Vec<Vec<T>>,
    context: C,
}

impl<I: PartialEq, L, F> MacroExpansion<I, Token<I, L>, F> {
    fn mk_macro_expansion_pos(&self, token: usize) -> Option<MacroExpansionPos> {
        if token >= self.def.body.len() {
            return None;
        }

        let param_expansion = self.def.body[token].name().and_then(|name| {
            self.param_position(name).map(|param| ParamExpansionPos {
                param,
                arg_token: 0,
            })
        });
        Some(MacroExpansionPos {
            token,
            param_expansion,
        })
    }

    fn param_position(&self, name: &I) -> Option<usize> {
        self.def.params.iter().position(|param| *param == *name)
    }

    fn next_pos(&self, pos: &MacroExpansionPos) -> Option<MacroExpansionPos> {
        let param_expansion = pos
            .param_expansion
            .as_ref()
            .and_then(|param_expansion| self.next_param_expansion_pos(&param_expansion));
        if param_expansion.is_some() {
            Some(MacroExpansionPos {
                param_expansion,
                ..*pos
            })
        } else {
            self.mk_macro_expansion_pos(pos.token + 1)
        }
    }

    fn next_param_expansion_pos(&self, pos: &ParamExpansionPos) -> Option<ParamExpansionPos> {
        if pos.arg_token + 1 < self.args[pos.param].len() {
            Some(ParamExpansionPos {
                arg_token: pos.arg_token + 1,
                ..*pos
            })
        } else {
            None
        }
    }

    fn token_and_span(&self, pos: MacroExpansionPos) -> (Token<I, L>, F::Span)
    where
        I: Clone,
        F: MacroCallCtx,
        Token<I, L>: Clone,
    {
        (self.token(&pos), self.context.mk_span(pos))
    }

    fn token(&self, pos: &MacroExpansionPos) -> Token<I, L>
    where
        I: Clone,
        Token<I, L>: Clone,
    {
        let body_token = &self.def.body[pos.token];
        pos.param_expansion.as_ref().map_or_else(
            || body_token.clone(),
            |param_expansion| match (
                body_token,
                &self.args[param_expansion.param][param_expansion.arg_token],
            ) {
                (Token::Label(_), Token::Ident(ident)) if param_expansion.arg_token == 0 => {
                    Token::Label(ident.clone())
                }
                (_, arg_token) => arg_token.clone(),
            },
        )
    }
}

impl<I, L> Token<I, L> {
    fn name(&self) -> Option<&I> {
        match &self {
            Token::Ident(name) | Token::Label(name) => Some(name),
            _ => None,
        }
    }
}

impl<I, L, F> MacroExpansionIter<I, Token<I, L>, F>
where
    I: PartialEq,
{
    fn new(
        def: Rc<MacroDefTokens<I, Token<I, L>>>,
        args: Vec<Vec<Token<I, L>>>,
        context: F,
    ) -> Self {
        let expansion = MacroExpansion { def, args, context };
        MacroExpansionIter {
            pos: expansion.mk_macro_expansion_pos(0),
            expansion,
        }
    }
}

impl<I, L, F> Iterator for MacroExpansionIter<I, Token<I, L>, F>
where
    I: Clone + PartialEq,
    F: MacroCallCtx,
    Token<I, L>: Clone,
{
    type Item = (Token<I, L>, F::Span);

    fn next(&mut self) -> Option<Self::Item> {
        self.pos.take().map(|pos| {
            self.pos = self.expansion.next_pos(&pos);
            self.expansion.token_and_span(pos)
        })
    }
}

#[cfg(test)]
pub mod mock {
    use super::*;

    use crate::analyze::syntax::Token;
    use crate::analyze::Literal;
    use crate::diag::DiagnosticsSystem;
    use crate::log::Log;

    #[derive(Debug, PartialEq)]
    pub(in crate::analyze) enum MacroTableEvent {
        DefineMacro(Vec<String>, Vec<Token<String, Literal<String>>>),
    }

    pub struct MockMacroTable<H, T> {
        macros: VecMacroTable<String, Literal<String>, H>,
        log: Log<T>,
    }

    impl<H, T> MockMacroTable<H, T> {
        pub fn new(log: Log<T>) -> Self {
            Self {
                macros: VecMacroTable::new(),
                log,
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct MockMacroId(pub usize);

    impl<H, T> MacroSource for MockMacroTable<H, T> {
        type MacroId = MockMacroId;
    }

    impl<'a, C, P, I, D, T> MacroSource
        for SourceComponents<
            C,
            P,
            &'a mut MockMacroTable<
                <D::Target as AddMacroDef<<D::Target as SpanSource>::Span>>::MacroDefHandle,
                T,
            >,
            I,
            D,
        >
    where
        D: DerefMut,
        D::Target: DiagnosticsSystem,
    {
        type MacroId = MockMacroId;
    }

    impl<'a, C, P, I, D, N, B, T>
        MacroTable<String, Literal<String>, <D::Target as SpanSource>::Span>
        for CompositeSession<
            SourceComponents<
                C,
                P,
                &'a mut MockMacroTable<
                    <D::Target as AddMacroDef<<D::Target as SpanSource>::Span>>::MacroDefHandle,
                    T,
                >,
                I,
                D,
            >,
            N,
            B,
        >
    where
        C: DerefMut,
        P: DerefMut,
        I: DerefMut,
        D: DerefMut,
        D::Target: DiagnosticsSystem,
        N: DerefMut,
        T: From<MacroTableEvent>,
    {
        type Iter = MacroExpansionIter<
            String,
            Token<String, Literal<String>>,
            <D::Target as MacroContextFactory<
                <D::Target as AddMacroDef<<D::Target as SpanSource>::Span>>::MacroDefHandle,
                <D::Target as SpanSource>::Span,
            >>::MacroCallCtx,
        >;

        fn define_macro(
            &mut self,
            name_span: <D::Target as SpanSource>::Span,
            params: (Vec<String>, Vec<<D::Target as SpanSource>::Span>),
            body: (
                Vec<Token<String, Literal<String>>>,
                Vec<<D::Target as SpanSource>::Span>,
            ),
        ) -> Self::MacroId {
            self.reentrancy
                .macros
                .log
                .push(MacroTableEvent::DefineMacro(
                    params.0.clone(),
                    body.0.clone(),
                ));
            MockMacroId(
                CompositeSession {
                    reentrancy: SourceComponents {
                        codebase: &mut *self.reentrancy.codebase,
                        parser_factory: &mut *self.reentrancy.parser_factory,
                        macros: &mut self.reentrancy.macros.macros,
                        interner: &mut *self.reentrancy.interner,
                        diagnostics: &mut *self.reentrancy.diagnostics,
                    },
                    names: &mut *self.names,
                    builder: (),
                }
                .define_macro(name_span, params, body)
                .0,
            )
        }

        fn expand_macro(
            &mut self,
            name: (Self::MacroId, <D::Target as SpanSource>::Span),
            args: MacroArgs<Token<String, Literal<String>>, <D::Target as SpanSource>::Span>,
        ) -> Self::Iter {
            CompositeSession {
                reentrancy: SourceComponents {
                    codebase: &mut *self.reentrancy.codebase,
                    parser_factory: &mut *self.reentrancy.parser_factory,
                    macros: &mut self.reentrancy.macros.macros,
                    interner: &mut *self.reentrancy.interner,
                    diagnostics: &mut *self.reentrancy.diagnostics,
                },
                names: &mut *self.names,
                builder: (),
            }
            .expand_macro((MacroId((name.0).0), name.1), args)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_macro_with_one_token() {
        let body = Token::<_, ()>::Ident("a");
        let macros = &mut vec![MacroDef {
            tokens: Rc::new(MacroDefTokens {
                params: vec![],
                body: vec![body.clone()],
            }),
            spans: (),
        }];
        let mut session = CompositeSession {
            reentrancy: SourceComponents {
                codebase: &mut (),
                parser_factory: &mut (),
                macros,
                interner: &mut (),
                diagnostics: &mut Factory,
            },
            names: &mut (),
            builder: (),
        };
        let name = ModularSpan::Buf(());
        let expanded: Vec<_> =
            MacroTable::expand_macro(&mut session, (MacroId(0), name.clone()), (vec![], vec![]))
                .collect();
        let data = MacroCall(Rc::new(ModularMacroCall {
            name,
            args: vec![],
            def: (),
        }));
        let macro_expansion_position = MacroExpansionPos {
            token: 0,
            param_expansion: None,
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

    // #[test]
    // fn expand_label_using_two_idents() {
    //     let label = Token::<_, ()>::Label("label");
    //     let macros = vec![MacroDef {
    //         tokens: Rc::new(MacroDefTokens {
    //             params: vec!["label"],
    //             body: vec![label],
    //         }),
    //         spans: (),
    //     }];
    //     let mut components = SourceComponents {
    //         codebase: &mut (),
    //         parser_factory: &mut (),
    //         macros,
    //         interner: &mut (),
    //         diagnostics: &mut Factory,
    //     };
    //     let name = ModularSpan::Buf(());
    //     let arg = vec![Token::Ident("tok1"), Token::Ident("tok2")];
    //     let expanded: Vec<_> = MacroTable::expand_macro(
    //         &mut components,
    //         (MacroId(0), name.clone()),
    //         (
    //             vec![arg],
    //             vec![vec![ModularSpan::Buf(()), ModularSpan::Buf(())]],
    //         ),
    //     )
    //     .collect();
    //     let context = MacroCall(Rc::new(ModularMacroCall {
    //         name,
    //         args: vec![vec![ModularSpan::Buf(()), ModularSpan::Buf(())]],
    //         def: (),
    //     }));
    //     let tok1_pos = MacroExpansionPos {
    //         token: 0,
    //         param_expansion: Some(ParamExpansionPos {
    //             param: 0,
    //             arg_token: 0,
    //         }),
    //     };
    //     let tok2_pos = MacroExpansionPos {
    //         token: 0,
    //         param_expansion: Some(ParamExpansionPos {
    //             param: 0,
    //             arg_token: 1,
    //         }),
    //     };
    //     assert_eq!(
    //         expanded,
    //         [
    //             (
    //                 Token::Label("tok1"),
    //                 ModularSpan::Macro(MacroSpan {
    //                     range: tok1_pos.clone()..=tok1_pos,
    //                     context: context.clone()
    //                 })
    //             ),
    //             (
    //                 Token::Ident("tok2"),
    //                 ModularSpan::Macro(MacroSpan {
    //                     range: tok2_pos.clone()..=tok2_pos,
    //                     context,
    //                 })
    //             )
    //         ]
    //     )
    // }

    // #[ignore]
    // #[test]
    // fn expand_macro() {
    //     let buf = Rc::new(BufContextData {
    //         buf_id: (),
    //         included_from: None,
    //     });
    //     let mk_span = |n| {
    //         ModularSpan::Buf(BufSpan {
    //             range: n,
    //             context: Rc::clone(&buf),
    //         })
    //     };
    //     let body: Vec<Token<_, ()>> = vec![Token::Ident("a"), Token::Ident("x"), Token::Ident("b")];
    //     let def_id = Rc::new(MacroDefSpans {
    //         name: mk_span(0),
    //         params: vec![mk_span(1)],
    //         body: (2..=4).map(mk_span).collect(),
    //     });
    //     let factory = &mut RcContextFactory::new();
    //     let entry = vec![MacroDef {
    //         tokens: Rc::new(MacroDefTokens {
    //             params: vec!["x"],
    //             body,
    //         }),
    //         spans: Rc::clone(&def_id),
    //     }];
    //     let data = RcMacroCall::new(ModularMacroCall {
    //         name: ModularSpan::Buf(BufSpan {
    //             range: 7,
    //             context: buf.clone(),
    //         }),
    //         args: vec![vec![
    //             ModularSpan::Buf(BufSpan {
    //                 range: 8,
    //                 context: buf.clone(),
    //             }),
    //             ModularSpan::Buf(BufSpan {
    //                 range: 9,
    //                 context: buf.clone(),
    //             }),
    //         ]],
    //         def: def_id,
    //     });
    //     let call_name = ("my_macro", mk_span(7));
    //     let expanded: Vec<_> = entry
    //         .expand_macro(
    //             (MacroId(0), call_name.1),
    //             (
    //                 vec![vec![Token::Ident("y"), Token::Ident("z")]],
    //                 vec![(8..=9).map(mk_span).collect()],
    //             ),
    //             factory,
    //         )
    //         .collect();
    //     let mk_span_data = |token, param_expansion| {
    //         let position = MacroExpansionPos {
    //             token,
    //             param_expansion,
    //         };
    //         ModularSpan::Macro(MacroSpan {
    //             range: position.clone()..=position,
    //             context: data.clone(),
    //         })
    //     };
    //     assert_eq!(
    //         expanded,
    //         [
    //             (Token::Ident("a"), mk_span_data(0, None)),
    //             (
    //                 Token::Ident("y"),
    //                 mk_span_data(
    //                     1,
    //                     Some(ParamExpansionPos {
    //                         param: 0,
    //                         arg_token: 0
    //                     })
    //                 ),
    //             ),
    //             (
    //                 Token::Ident("z"),
    //                 mk_span_data(
    //                     1,
    //                     Some(ParamExpansionPos {
    //                         param: 0,
    //                         arg_token: 1
    //                     })
    //                 ),
    //             ),
    //             (Token::Ident("b"), mk_span_data(2, None)),
    //         ]
    //     )
    // }

    #[derive(Clone, Debug, PartialEq)]
    struct MacroCall(Rc<ModularMacroCall<(), Span>>);

    type Span = ModularSpan<(), MacroSpan<MacroCall>>;

    struct Factory;

    impl SpanSource for Factory {
        type Span = Span;
    }

    impl AddMacroDef<Span> for Factory {
        type MacroDefHandle = ();

        fn add_macro_def<P, B>(&mut self, _: Span, _: P, _: B) -> Self::MacroDefHandle
        where
            P: IntoIterator<Item = Span>,
            B: IntoIterator<Item = Span>,
        {
        }
    }

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
        fn mk_span(&self, position: MacroExpansionPos) -> Self::Span {
            ModularSpan::Macro(MacroSpan {
                range: position.clone()..=position,
                context: self.clone(),
            })
        }
    }
}
