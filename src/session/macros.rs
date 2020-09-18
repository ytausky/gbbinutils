use super::lex::{Lex, Literal, StringSource};
use super::NextToken;

use crate::codebase::Codebase;
use crate::semantics::{Semantics, TokenStreamState};
use crate::session::builder::Backend;
use crate::session::diagnostics::EmitDiag;
use crate::session::lex::LexItem;
use crate::session::resolve::{NameTable, StartScope};
use crate::session::{Interner, TokenStream};
use crate::span::*;
use crate::syntax::parser::{DefaultParserFactory, ParseTokenStream, ParserFactory};
use crate::syntax::LexError;
use crate::syntax::Token;
use crate::CompositeSession;

use std::fmt::Debug;
use std::rc::Rc;

pub(crate) trait MacroSource {
    type MacroId: Clone;
}

pub(crate) trait MacroTable<I, L, S: Clone>: MacroSource {
    fn define_macro(
        &mut self,
        name_span: S,
        params: Box<[(I, S)]>,
        body: Box<[(Token<I, L>, S)]>,
    ) -> Self::MacroId;

    fn expand_macro(&mut self, name: (Self::MacroId, S), args: MacroArgs<Token<I, L>, S>);
}

pub(crate) type VecMacroTable<R, S> = Vec<Rc<MacroDef<R, S>>>;

pub type MacroArgs<T, S> = Box<[Box<[(T, S)]>]>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacroId(usize);

impl<'a, C, R: SpanSource, II: StringSource, N, B, D, RR, S> MacroSource
    for CompositeSession<C, R, II, VecMacroTable<RR, S>, N, B, D>
{
    type MacroId = MacroId;
}

impl<'a, C, RR, II, N, B, D>
    MacroTable<II::StringRef, Literal<II::StringRef>, <Self as SpanSource>::Span>
    for CompositeSession<C, RR, II, VecMacroTable<II::StringRef, <RR as SpanSource>::Span>, N, B, D>
where
    Self: Lex<RR, II, Span = RR::Span, StringRef = II::StringRef>,
    C: Codebase,
    RR: SpanSystem<II::StringRef>,
    II: Interner,
    Self: NextToken,
    Self: Interner,
    Self: EmitDiag<RR::Span, RR::Stripped>,
    Self: StartScope<II::StringRef> + NameTable<II::StringRef>,
    Self: Backend<RR::Span>,
    Self: MacroSource<MacroId = MacroId>,
    <Self as StringSource>::StringRef: 'static,
    <Self as SpanSource>::Span: 'static,
    <Self as Lex<RR, II>>::TokenIter: 'static,
{
    fn define_macro(
        &mut self,
        name_span: RR::Span,
        params: Box<[(II::StringRef, RR::Span)]>,
        body: Box<[(Token<II::StringRef, Literal<II::StringRef>>, RR::Span)]>,
    ) -> Self::MacroId {
        let id = MacroId(self.macros.len());
        self.macros.push(Rc::new(MacroDef {
            name_span,
            params,
            body,
        }));
        id
    }

    fn expand_macro(
        &mut self,
        (MacroId(id), name_span): (Self::MacroId, RR::Span),
        args: MacroArgs<Token<II::StringRef, Literal<II::StringRef>>, RR::Span>,
    ) {
        let def = &self.macros[id];
        let expansion = MacroExpansionIter::new(name_span, Rc::clone(def), args);
        self.tokens.push(Box::new(expansion));
        let mut parser = <DefaultParserFactory as ParserFactory<
            II::StringRef,
            Literal<II::StringRef>,
            LexError,
            RR::Span,
        >>::mk_parser(&mut DefaultParserFactory);
        let semantics = Semantics {
            session: self,
            state: TokenStreamState::new(),
        };
        parser.parse_token_stream(semantics);
    }
}

pub struct MacroExpansionIter<R, S> {
    expansion: Rc<MacroExpansion<R, S>>,
    pos: Option<MacroExpansionPos>,
}

impl<R: Clone + PartialEq, S> MacroExpansion<R, S> {
    fn mk_macro_expansion_pos(&self, token: usize) -> Option<MacroExpansionPos> {
        if token >= self.def.body.len() {
            return None;
        }

        let param_expansion = self.def.body[token].0.name().and_then(|name| {
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

    fn param_position(&self, name: &R) -> Option<usize> {
        self.def
            .params
            .iter()
            .position(|(param, _)| *param == *name)
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
}

impl<R, S> MacroExpansionIter<R, S>
where
    R: Clone,
    S: Clone,
{
    fn token_and_span<RR>(
        &self,
        pos: MacroExpansionPos,
        registry: &mut RR,
    ) -> (Token<R, Literal<R>>, RR::Span)
    where
        RR: SpanSystem<R, Span = S>,
    {
        (
            self.token(&pos),
            registry.encode_span(SpanDraft::MacroExpansion(
                Rc::clone(&self.expansion),
                pos.clone()..=pos,
            )),
        )
    }

    fn token(&self, pos: &MacroExpansionPos) -> Token<R, Literal<R>> {
        let body_token = &self.expansion.def.body[pos.token].0;
        pos.param_expansion.as_ref().map_or_else(
            || body_token.clone(),
            |param_expansion| match (
                body_token,
                &self.expansion.args[param_expansion.param][param_expansion.arg_token].0,
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

impl<R: Clone + PartialEq, S> MacroExpansionIter<R, S> {
    fn new(
        name_span: S,
        def: Rc<MacroDef<R, S>>,
        args: Box<[Box<[(Token<R, Literal<R>>, S)]>]>,
    ) -> Self {
        let expansion = Rc::new(MacroExpansion {
            name_span,
            def,
            args,
        });
        MacroExpansionIter {
            pos: expansion.mk_macro_expansion_pos(0),
            expansion,
        }
    }
}

impl<RR, II, S> TokenStream<RR, II> for MacroExpansionIter<II::StringRef, S>
where
    RR: SpanSystem<II::StringRef, Span = S>,
    II: StringSource,
    S: Clone,
{
    fn next_token(
        &mut self,
        registry: &mut RR,
        _interner: &mut II,
    ) -> Option<LexItem<II::StringRef, RR::Span>> {
        self.pos.take().map(|pos| {
            self.pos = self.expansion.next_pos(&pos);
            let (token, span) = self.token_and_span(pos, registry);
            (Ok(token), span)
        })
    }
}

#[cfg(test)]
pub mod mock {
    use super::*;

    use crate::log::Log;
    use crate::session::lex::Literal;
    use crate::syntax::Token;

    #[derive(Debug, PartialEq)]
    pub enum MacroTableEvent {
        DefineMacro(Box<[String]>, Box<[Token<String, Literal<String>>]>),
        ExpandMacro(MockMacroId, Box<[Box<[Token<String, Literal<String>>]>]>),
    }

    pub struct MockMacroTable<T> {
        log: Log<T>,
    }

    impl<T> MockMacroTable<T> {
        pub fn new(log: Log<T>) -> Self {
            Self { log }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct MockMacroId(pub usize);

    impl<T> MacroSource for MockMacroTable<T> {
        type MacroId = MockMacroId;
    }

    impl<'a, C, R: SpanSource, I: StringSource, N, B, D, T> MacroSource
        for CompositeSession<C, R, I, MockMacroTable<T>, N, B, D>
    {
        type MacroId = MockMacroId;
    }

    impl<C, R, I: StringSource, N, B, D, T> MacroTable<String, Literal<String>, D::Span>
        for CompositeSession<C, R, I, MockMacroTable<T>, N, B, D>
    where
        R: SpanSource,
        D: SpanSource,
        T: From<MacroTableEvent>,
    {
        fn define_macro(
            &mut self,
            _name_span: D::Span,
            params: Box<[(String, D::Span)]>,
            body: Box<[(Token<String, Literal<String>>, D::Span)]>,
        ) -> Self::MacroId {
            self.macros.log.push(MacroTableEvent::DefineMacro(
                Vec::from(params)
                    .into_iter()
                    .map(|(param, _)| param)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                Vec::from(body)
                    .into_iter()
                    .map(|(token, _)| token)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ));
            MockMacroId(0)
        }

        fn expand_macro(
            &mut self,
            name: (Self::MacroId, D::Span),
            args: MacroArgs<Token<String, Literal<String>>, D::Span>,
        ) {
            let args = Vec::from(args)
                .into_iter()
                .map(|arg| {
                    Vec::from(arg)
                        .into_iter()
                        .map(|(arg, _)| arg)
                        .collect::<Vec<_>>()
                        .into_boxed_slice()
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            self.macros
                .log
                .push(MacroTableEvent::ExpandMacro(name.0, args))
        }
    }
}

#[cfg(test)]
mod tests {
    // #[test]
    // fn expand_macro_with_one_token() {
    //     let body = Token::<_, ()>::Ident("a");
    //     let macros = vec![MacroDef {
    //         tokens: Rc::new(MacroDefTokens {
    //             params: vec![],
    //             body: vec![body.clone()],
    //         }),
    //         spans: (),
    //     }];
    //     let mut session = CompositeSession {
    //         reentrancy: SourceComponents {
    //             codebase: (),
    //             parser_factory: (),
    //             macros,
    //             interner: (),
    //         },
    //         names: (),
    //         builder: (),
    //         diagnostics: Factory,
    //     };
    //     let name = ModularSpan::Buf(());
    //     // let expanded: Vec<_> =
    //     //     MacroTable::expand_macro(&mut session, (MacroId(0), name.clone()), (vec![], vec![]))
    //     //         .collect();
    //     let data = MacroCall(Rc::new(ModularMacroCall {
    //         name,
    //         args: vec![],
    //         def: (),
    //     }));
    //     let macro_expansion_position = MacroExpansionPos {
    //         token: 0,
    //         param_expansion: None,
    //     };
    //     assert_eq!(
    //         expanded,
    //         [(
    //             body,
    //             ModularSpan::Macro(MacroSpan {
    //                 range: macro_expansion_position.clone()..=macro_expansion_position,
    //                 context: data,
    //             })
    //         )]
    //     )
    // }

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
}
