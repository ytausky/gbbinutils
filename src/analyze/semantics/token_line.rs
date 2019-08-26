use super::{Keyword, Label, Session, TokenStreamSemantics, TokenStreamState};

use crate::analyze::resolve::{NameTable, ResolvedName, StartScope};
use crate::analyze::session::ReentrancyActions;
use crate::analyze::syntax::actions::{LineFinalizer, TokenLineActions, TokenLineRule};
use crate::analyze::syntax::{Sigil, Token};
use crate::analyze::{Literal, SemanticToken, TokenSeq};
use crate::object::builder::SymbolSource;

use std::ops::DerefMut;

pub(in crate::analyze) type TokenLineSemantics<R, N, B> = Session<R, N, B, TokenContext<R>>;

pub(in crate::analyze) enum TokenContext<S: ReentrancyActions> {
    MacroDef(MacroDefState<S>),
}

pub(in crate::analyze) struct MacroDefState<S: ReentrancyActions> {
    label: Option<Label<S::Ident, S::Span>>,
    tokens: TokenSeq<S::Ident, S::StringRef, S::Span>,
}

impl<S: ReentrancyActions> MacroDefState<S> {
    pub(super) fn new(label: Option<Label<S::Ident, S::Span>>) -> Self {
        Self {
            label,
            tokens: (Vec::new(), Vec::new()),
        }
    }
}

impl<R, N, B> TokenLineActions<R::Ident, Literal<R::StringRef>, R::Span>
    for TokenLineSemantics<R, N, B>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: SymbolSource,
{
    type ContextFinalizer = TokenContextFinalizationSemantics<R, N, B>;

    fn act_on_token(&mut self, token: SemanticToken<R::Ident, R::StringRef>, span: R::Span) {
        match &mut self.state {
            TokenContext::MacroDef(state) => {
                state.tokens.0.push(token);
                state.tokens.1.push(span);
            }
        }
    }

    fn act_on_ident(
        mut self,
        ident: R::Ident,
        span: R::Span,
    ) -> TokenLineRule<Self, Self::ContextFinalizer> {
        match &mut self.state {
            TokenContext::MacroDef(state) => {
                if ident.as_ref().eq_ignore_ascii_case("ENDM") {
                    state.tokens.0.push(Sigil::Eos.into());
                    state.tokens.1.push(span);
                    TokenLineRule::LineEnd(TokenContextFinalizationSemantics { parent: self })
                } else {
                    state.tokens.0.push(Token::Ident(ident));
                    state.tokens.1.push(span);
                    TokenLineRule::TokenSeq(self)
                }
            }
        }
    }
}

impl<R: ReentrancyActions, N, B> LineFinalizer<R::Span> for TokenLineSemantics<R, N, B> {
    type Next = TokenStreamSemantics<R, N, B>;

    fn did_parse_line(mut self, span: R::Span) -> Self::Next {
        match &mut self.state {
            TokenContext::MacroDef(state) => {
                state.tokens.0.push(Sigil::Eol.into());
                state.tokens.1.push(span);
            }
        }
        set_state!(self, self.state.into())
    }
}

pub(in crate::analyze) struct TokenContextFinalizationSemantics<R: ReentrancyActions, N, B> {
    parent: TokenLineSemantics<R, N, B>,
}

delegate_diagnostics! {
    {R: ReentrancyActions, N, B}, TokenContextFinalizationSemantics<R, N, B>, {parent}, R, R::Span
}

impl<R, N, B> LineFinalizer<R::Span> for TokenContextFinalizationSemantics<R, N, B>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: NameTable<R::Ident, MacroId = R::MacroId, SymbolId = B::SymbolId>,
    B: SymbolSource,
{
    type Next = TokenStreamSemantics<R, N, B>;

    fn did_parse_line(mut self, _: R::Span) -> Self::Next {
        match self.parent.state {
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
