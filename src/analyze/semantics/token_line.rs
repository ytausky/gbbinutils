use super::{Keyword, Label, Session, TokenStreamSemantics};

use crate::analyze::resolve::ResolvedName;
use crate::analyze::session::ReentrancyActions;
use crate::analyze::syntax::actions::{LineFinalizer, TokenLineActions, TokenLineRule};
use crate::analyze::syntax::{Sigil, Token};
use crate::analyze::{Literal, SemanticToken, TokenSeq};

pub(in crate::analyze) type TokenLineSemantics<S> = Session<S, TokenContext<S>>;

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

impl<S> TokenLineActions<S::Ident, Literal<S::StringRef>, S::Span> for TokenLineSemantics<S>
where
    S: ReentrancyActions<Keyword = &'static Keyword>,
{
    type ContextFinalizer = TokenContextFinalizationSemantics<S>;

    fn act_on_token(&mut self, token: SemanticToken<S::Ident, S::StringRef>, span: S::Span) {
        match &mut self.state {
            TokenContext::MacroDef(state) => {
                state.tokens.0.push(token);
                state.tokens.1.push(span);
            }
        }
    }

    fn act_on_ident(
        mut self,
        ident: S::Ident,
        span: S::Span,
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

impl<S: ReentrancyActions> LineFinalizer<S::Span> for TokenLineSemantics<S> {
    type Next = TokenStreamSemantics<S>;

    fn did_parse_line(mut self, span: S::Span) -> Self::Next {
        match &mut self.state {
            TokenContext::MacroDef(state) => {
                state.tokens.0.push(Sigil::Eol.into());
                state.tokens.1.push(span);
            }
        }
        set_state!(self, self.state.into())
    }
}

pub(in crate::analyze) struct TokenContextFinalizationSemantics<S: ReentrancyActions> {
    parent: TokenLineSemantics<S>,
}

delegate_diagnostics! {
    {S: ReentrancyActions}, TokenContextFinalizationSemantics<S>, {parent}, S, S::Span
}

impl<S: ReentrancyActions<Keyword = &'static Keyword>> LineFinalizer<S::Span>
    for TokenContextFinalizationSemantics<S>
{
    type Next = TokenStreamSemantics<S>;

    fn did_parse_line(mut self, _: S::Span) -> Self::Next {
        match self.parent.state {
            TokenContext::MacroDef(state) => {
                if let Some((name, params)) = state.label {
                    let tokens = state.tokens;
                    let id = self.parent.reentrancy.define_macro(name.1, params, tokens);
                    self.parent
                        .reentrancy
                        .insert(name.0, ResolvedName::Macro(id));
                }
            }
        }
        TokenStreamSemantics::new(self.parent.reentrancy)
    }
}
