use super::lex::Lex;
use super::{CompositeSession, MacroSource, NextToken, StringSource};

use crate::assembler::semantics::{Semantics, TokenStreamState};
use crate::assembler::session::resolve::StartScope;
use crate::assembler::session::{Backend, Interner, NameTable, TokenStream};
use crate::assembler::syntax::parser::{DefaultParserFactory, ParseTokenStream, ParserFactory};
use crate::assembler::syntax::{LexError, LexItem, Literal, Token};
use crate::codebase::{BufId, Codebase};
use crate::diagnostics::EmitDiag;
use crate::span::*;

use std::fmt::Debug;
use std::rc::Rc;

pub(in crate::assembler) trait MacroTable<I, L, S: Clone>:
    MacroSource
{
    fn define_macro(
        &mut self,
        name_span: S,
        params: (Box<[I]>, Box<[S]>),
        body: (Box<[Token<I, L>]>, Box<[S]>),
    ) -> Self::MacroId;

    fn expand_macro(&mut self, name: (Self::MacroId, S), args: MacroArgs<Token<I, L>, S>);
}

pub(crate) type VecMacroTable<D, R> = Vec<Rc<MacroDef<D, R>>>;

pub type MacroArgs<T, S> = (Box<[Box<[T]>]>, Box<[Box<[S]>]>);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacroId(usize);

impl<D, R> MacroSource for VecMacroTable<D, R> {
    type MacroId = MacroId;
}

impl<'a, C, R: SpanSource, II: StringSource, M: MacroSource, N, B, D> MacroSource
    for CompositeSession<C, R, II, M, N, B, D>
{
    type MacroId = M::MacroId;
}

impl<'a, C, R, I, N, B, D>
    MacroTable<I::StringRef, Literal<I::StringRef>, <Self as SpanSource>::Span>
    for CompositeSession<C, R, I, VecMacroTable<R::MacroDefMetadataId, I::StringRef>, N, B, D>
where
    Self: Lex<R, I, Span = R::Span, StringRef = I::StringRef>,
    C: Codebase,
    R: SpanSystem<BufId>,
    I: Interner,
    Self: NextToken,
    Self: Interner,
    Self: EmitDiag<R::Span, R::Stripped>,
    Self: StartScope + NameTable<I::StringRef>,
    Self: Backend<R::Span>,
    Self: MacroSource<MacroId = MacroId>,
    <Self as StringSource>::StringRef: 'static,
    <Self as SpanSource>::Span: 'static,
    <Self as Lex<R, I>>::TokenIter: 'static,
{
    fn define_macro(
        &mut self,
        name_span: R::Span,
        (params, param_spans): (Box<[I::StringRef]>, Box<[R::Span]>),
        (body, body_spans): (
            Box<[Token<I::StringRef, Literal<I::StringRef>>]>,
            Box<[R::Span]>,
        ),
    ) -> Self::MacroId {
        let id = MacroId(self.macros.len());
        let metadata = self.registry.add_macro_def(MacroDefMetadata {
            name_span,
            param_spans,
            body_spans,
        });
        self.macros.push(Rc::new(MacroDef {
            metadata,
            params,
            body,
        }));
        id
    }

    fn expand_macro(
        &mut self,
        (MacroId(id), name_span): (Self::MacroId, R::Span),
        (args, arg_spans): MacroArgs<Token<I::StringRef, Literal<I::StringRef>>, R::Span>,
    ) {
        let def = &self.macros[id];
        let metadata = self.registry.add_macro_expansion(MacroExpansionMetadata {
            def: def.metadata.clone(),
            name_span,
            arg_spans,
        });
        let expansion = MacroExpansionIter::new(metadata, Rc::clone(def), args);
        self.tokens.push(Box::new(expansion));
        let mut parser = <DefaultParserFactory as ParserFactory<
            I::StringRef,
            Literal<I::StringRef>,
            LexError,
            R::Span,
        >>::mk_parser(&mut DefaultParserFactory);
        let semantics = Semantics {
            session: self,
            state: TokenStreamState::new(),
        };
        parser.parse_token_stream(semantics);
    }
}

pub struct MacroExpansionIter<E, D, R> {
    metadata: E,
    def: Rc<MacroDef<D, R>>,
    args: Box<[Box<[Token<R, Literal<R>>]>]>,
    pos: Option<MacroExpansionPos>,
}

#[derive(Debug)]
pub(crate) struct MacroExpansion<E, D, R> {
    metadata: E,
    def: Rc<MacroDef<D, R>>,
    args: Box<[Box<[Token<R, Literal<R>>]>]>,
}

#[derive(Debug)]
pub(crate) struct MacroDef<D, R> {
    metadata: D,
    params: Box<[R]>,
    body: Box<[Token<R, Literal<R>>]>,
}

impl<D, R: Clone + PartialEq> MacroDef<D, R> {
    fn mk_macro_expansion_pos(&self, token: usize) -> Option<MacroExpansionPos> {
        if token >= self.body.len() {
            return None;
        }

        let param_expansion = self.body[token].name().and_then(|name| {
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
        self.params.iter().position(|param| *param == *name)
    }
}

impl<R: Clone + PartialEq, S> MacroExpansion<Token<R, Literal<R>>, R, S> {}

impl<E: Clone + 'static, D, R: Clone + PartialEq> MacroExpansionIter<E, D, R> {
    fn token_and_span<RR>(
        &self,
        pos: MacroExpansionPos,
        registry: &mut RR,
    ) -> (Token<R, Literal<R>>, RR::Span)
    where
        RR: SpanSystem<BufId, MacroExpansionMetadataId = E>,
    {
        (
            self.token(&pos),
            registry.encode_span(Span::MacroExpansion(
                self.metadata.clone(),
                pos.clone()..=pos,
            )),
        )
    }

    fn token(&self, pos: &MacroExpansionPos) -> Token<R, Literal<R>> {
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
            self.def.mk_macro_expansion_pos(pos.token + 1)
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

impl<I, L> Token<I, L> {
    fn name(&self) -> Option<&I> {
        match &self {
            Token::Ident(name) | Token::Label(name) => Some(name),
            _ => None,
        }
    }
}

impl<E, D, R: Clone + PartialEq> MacroExpansionIter<E, D, R> {
    fn new(metadata: E, def: Rc<MacroDef<D, R>>, args: Box<[Box<[Token<R, Literal<R>>]>]>) -> Self {
        let pos = def.mk_macro_expansion_pos(0);
        MacroExpansionIter {
            metadata,
            def,
            args,
            pos,
        }
    }
}

impl<R, I> TokenStream<R, I>
    for MacroExpansionIter<R::MacroExpansionMetadataId, R::MacroDefMetadataId, I::StringRef>
where
    R: SpanSystem<BufId>,
    I: StringSource,
{
    fn next_token(
        &mut self,
        registry: &mut R,
        _interner: &mut I,
    ) -> Option<LexItem<I::StringRef, R::Span>> {
        self.pos.take().map(|pos| {
            self.pos = self.next_pos(&pos);
            let (token, span) = self.token_and_span(pos, registry);
            (Ok(token), span)
        })
    }
}

#[cfg(test)]
pub mod mock {
    use super::*;

    use crate::assembler::session::mock::{MacroTableEvent, MockMacroId};
    use crate::assembler::syntax::{Literal, Token};
    use crate::log::Log;

    pub struct MockMacroTable<T> {
        log: Log<T>,
    }

    impl<T> MockMacroTable<T> {
        pub fn new(log: Log<T>) -> Self {
            Self { log }
        }
    }

    impl<T> MacroSource for MockMacroTable<T> {
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
            (params, _): (Box<[String]>, Box<[D::Span]>),
            (body, _): (Box<[Token<String, Literal<String>>]>, Box<[D::Span]>),
        ) -> Self::MacroId {
            self.macros
                .log
                .push(MacroTableEvent::DefineMacro(params, body));
            MockMacroId(0)
        }

        fn expand_macro(
            &mut self,
            name: (Self::MacroId, D::Span),
            args: MacroArgs<Token<String, Literal<String>>, D::Span>,
        ) {
            self.macros
                .log
                .push(MacroTableEvent::ExpandMacro(name.0, args.0))
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
