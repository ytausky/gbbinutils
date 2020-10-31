use super::*;

use crate::assembler::semantics::SemanticActions;
use crate::assembler::session::resolve::StartScope;
use crate::assembler::session::{Backend, NameTable, TokenStream};
use crate::assembler::syntax::parser::{DefaultParserFactory, ParseTokenStream, ParserFactory};
use crate::assembler::syntax::{LexError, LexItem, Literal, SemanticToken, Token};
use crate::diagnostics::EmitDiag;
use crate::object::*;

use std::fmt::Debug;
use std::rc::Rc;

pub(in crate::assembler) trait MacroTable<S: Clone> {
    fn define_macro(
        &mut self,
        name: (StringRef, S),
        params: (Box<[StringRef]>, Box<[S]>),
        body: (Box<[SemanticToken]>, Box<[S]>),
    );

    fn expand_macro(&mut self, name: (MacroId, S), args: MacroArgs<S>);
}

pub(crate) type VecMacroTable = Vec<Rc<MacroDef>>;

pub type MacroArgs<S> = (Box<[Box<[SemanticToken]>]>, Box<[Box<[S]>]>);

impl<'a, R> MacroTable<R::Span> for CompositeSession<'a, R>
where
    R: SpanSystem,
    Self: SpanSource<Span = R::Span>,
    Self: NextToken,
    Self: EmitDiag<R::Span, R::Stripped>,
    Self: StartScope + NameTable<StringRef>,
    Self: Backend<R::Span>,
    for<'r> DiagnosticsContext<'r, 'a, R, OutputForwarder<'a>>: EmitDiag<R::Span, R::Stripped>,
    R::Span: 'static,
    R::Stripped: Clone,
{
    fn define_macro(
        &mut self,
        (name, name_span): (StringRef, R::Span),
        (params, param_spans): (Box<[StringRef]>, Box<[R::Span]>),
        (body, body_spans): (Box<[SemanticToken]>, Box<[R::Span]>),
    ) {
        #[cfg(test)]
        self.log_event(Event::DefineMacro {
            name: (name.clone(), name_span.clone()),
            params: (params.clone(), param_spans.clone()),
            body: (body.clone(), body_spans.clone()),
        });

        let id = MacroId(self.macros.len());
        let metadata = self.metadata.add_macro_def(MacroDefMetadata {
            name_span,
            param_spans,
            body_spans,
        });
        self.macros.push(Rc::new(MacroDef {
            metadata,
            params,
            body,
        }));
        self.mnemonics.insert(name, MnemonicEntry::Macro(id));
    }

    fn expand_macro(
        &mut self,
        (MacroId(id), name_span): (MacroId, R::Span),
        (args, arg_spans): MacroArgs<R::Span>,
    ) {
        #[cfg(test)]
        self.log_event(Event::ExpandMacro {
            name: (MacroId(id), name_span.clone()),
            args: (args.clone(), arg_spans.clone()),
        });

        let def = &self.macros[id];
        let metadata = self.metadata.add_macro_expansion(MacroExpansionMetadata {
            def: def.metadata,
            name_span,
            arg_spans,
        });
        let expansion = MacroExpansionIter::new(metadata, Rc::clone(def), args);
        self.tokens.push(Box::new(expansion));
        let mut parser = <DefaultParserFactory as ParserFactory<
            StringRef,
            Literal,
            LexError,
            R::Span,
        >>::mk_parser(&mut DefaultParserFactory);
        parser.parse_token_stream(self.semantic_actions());
    }
}

pub struct MacroExpansionIter {
    metadata: MacroExpansionId,
    def: Rc<MacroDef>,
    args: Box<[Box<[SemanticToken]>]>,
    pos: Option<MacroExpansionPos>,
}

#[derive(Debug)]
pub(crate) struct MacroExpansion<E> {
    metadata: E,
    def: Rc<MacroDef>,
    args: Box<[Box<[SemanticToken]>]>,
}

#[derive(Debug)]
pub(crate) struct MacroDef {
    metadata: MacroDefId,
    params: Box<[StringRef]>,
    body: Box<[SemanticToken]>,
}

impl MacroDef {
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

    fn param_position(&self, name: &str) -> Option<usize> {
        self.params.iter().position(|param| **param == *name)
    }
}

impl MacroExpansionIter {
    fn token_and_span<RR>(
        &self,
        pos: MacroExpansionPos,
        registry: &mut RR,
    ) -> (SemanticToken, RR::Span)
    where
        RR: SpanSystem,
    {
        (
            self.token(&pos),
            registry.encode_span(Span::MacroExpansion {
                metadata: self.metadata,
                range: pos.clone()..=pos,
            }),
        )
    }

    fn token(&self, pos: &MacroExpansionPos) -> SemanticToken {
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

impl MacroExpansionIter {
    fn new(
        metadata: MacroExpansionId,
        def: Rc<MacroDef>,
        args: Box<[Box<[SemanticToken]>]>,
    ) -> Self {
        let pos = def.mk_macro_expansion_pos(0);
        MacroExpansionIter {
            metadata,
            def,
            args,
            pos,
        }
    }
}

impl<R> TokenStream<R> for MacroExpansionIter
where
    R: SpanSystem,
{
    fn next_token(&mut self, registry: &mut R) -> Option<LexItem<R::Span>> {
        self.pos.take().map(|pos| {
            self.pos = self.next_pos(&pos);
            let (token, span) = self.token_and_span(pos, registry);
            (Ok(token), span)
        })
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
