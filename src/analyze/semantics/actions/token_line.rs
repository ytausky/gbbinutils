use super::{Keyword, TokenStreamSemantics};

use crate::analyze::reentrancy::ReentrancyActions;
use crate::analyze::semantics::builtin_instr::Dispatch;
use crate::analyze::semantics::resolve::{NameTable, ResolvedName, StartScope};
use crate::analyze::semantics::*;
use crate::analyze::syntax::actions::{LineFinalizer, TokenLineActions, TokenLineRule};
use crate::analyze::syntax::{Sigil, Token};
use crate::analyze::{Literal, SemanticToken};
use crate::object::builder::SymbolSource;

use std::ops::DerefMut;

impl<I, R, N, B> TokenLineActions<R::Ident, Literal<R::StringRef>, R::Span>
    for TokenLineSemantics<I, R, N, B>
where
    I: BuiltinInstrSet<R>,
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword<I::Binding, I::Free>,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: SymbolSource,
    BuiltinInstr<&'static I::Binding, &'static I::Free, R>: Dispatch<I, R>,
{
    type ContextFinalizer = TokenContextFinalizationSemantics<I, R, N, B>;

    fn act_on_token(&mut self, token: SemanticToken<R::Ident, R::StringRef>, span: R::Span) {
        match &mut self.state.context {
            TokenContext::FalseIf => (),
            TokenContext::MacroDef(state) => state.act_on_token(token, span),
        }
    }

    fn act_on_mnemonic(
        mut self,
        ident: R::Ident,
        span: R::Span,
    ) -> TokenLineRule<Self, Self::ContextFinalizer> {
        match &mut self.state.context {
            TokenContext::FalseIf => {
                if ident.as_ref().eq_ignore_ascii_case("ENDC") {
                    TokenLineRule::LineEnd(TokenContextFinalizationSemantics { parent: self })
                } else {
                    TokenLineRule::TokenSeq(self)
                }
            }
            TokenContext::MacroDef(state) => {
                if ident.as_ref().eq_ignore_ascii_case("ENDM") {
                    state.act_on_token(Sigil::Eos.into(), span);
                    TokenLineRule::LineEnd(TokenContextFinalizationSemantics { parent: self })
                } else {
                    state.act_on_token(Token::Ident(ident), span);
                    TokenLineRule::TokenSeq(self)
                }
            }
        }
    }
}

impl<I, R, S> MacroDefState<I, R, S> {
    fn act_on_token(&mut self, token: SemanticToken<I, R>, span: S) {
        self.tokens.0.push(token);
        self.tokens.1.push(span);
    }
}

impl<I, R: ReentrancyActions, N, B> LineFinalizer<R::Span> for TokenLineSemantics<I, R, N, B> {
    type Next = TokenStreamSemantics<I, R, N, B>;

    fn did_parse_line(mut self, span: R::Span) -> Self::Next {
        match &mut self.state.context {
            TokenContext::FalseIf => (),
            TokenContext::MacroDef(state) => state.act_on_token(Sigil::Eol.into(), span),
        }
        set_state!(self, self.state.into())
    }
}

pub(in crate::analyze) struct TokenContextFinalizationSemantics<I, R: ReentrancyActions, N, B> {
    parent: TokenLineSemantics<I, R, N, B>,
}

delegate_diagnostics! {
    {I, R: ReentrancyActions, N, B},
    TokenContextFinalizationSemantics<I, R, N, B>,
    {parent},
    R,
    R::Span
}

impl<I, R, N, B> LineFinalizer<R::Span> for TokenContextFinalizationSemantics<I, R, N, B>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: NameTable<R::Ident, MacroId = R::MacroId, SymbolId = B::SymbolId>,
    B: SymbolSource,
{
    type Next = TokenStreamSemantics<I, R, N, B>;

    fn did_parse_line(mut self, _: R::Span) -> Self::Next {
        match self.parent.state.context {
            TokenContext::FalseIf => (),
            TokenContext::MacroDef(state) => {
                if let Some((name, params)) = state.label {
                    let tokens = state.tokens;
                    let id = self.parent.reentrancy.define_macro(name.1, params, tokens);
                    self.parent
                        .names
                        .define_name(name.0, ResolvedName::Macro(id));
                }
            }
        }
        set_state!(self.parent, TokenStreamState::new())
    }
}
