use super::{Label, SemanticActions, TokenStreamSemantics};

use crate::analyze::resolve::ResolvedName;
use crate::analyze::session::Session;
use crate::analyze::syntax::actions::{LineFinalizer, TokenLineActions, TokenLineRule};
use crate::analyze::syntax::{Sigil, Token};
use crate::analyze::{Literal, SemanticToken, TokenSeq};

pub(in crate::analyze) type TokenLineSemantics<S> = SemanticActions<TokenContext<S>, S>;

pub(in crate::analyze) enum TokenContext<S: Session> {
    MacroDef(MacroDefState<S>),
}

pub(in crate::analyze) struct MacroDefState<S: Session> {
    label: Option<Label<S::Ident, S::Span>>,
    tokens: TokenSeq<S::Ident, S::StringRef, S::Span>,
}

impl<S: Session> MacroDefState<S> {
    pub(super) fn new(label: Option<Label<S::Ident, S::Span>>) -> Self {
        Self {
            label,
            tokens: (Vec::new(), Vec::new()),
        }
    }
}

impl<S: Session> TokenLineActions<S::Ident, Literal<S::StringRef>, S::Span>
    for TokenLineSemantics<S>
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

impl<S: Session> LineFinalizer<S::Span> for TokenLineSemantics<S> {
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

pub(in crate::analyze) struct TokenContextFinalizationSemantics<S: Session> {
    parent: TokenLineSemantics<S>,
}

delegate_diagnostics! {
    {S: Session}, TokenContextFinalizationSemantics<S>, {parent}, S, S::Span
}

impl<S: Session> LineFinalizer<S::Span> for TokenContextFinalizationSemantics<S> {
    type Next = TokenStreamSemantics<S>;

    fn did_parse_line(mut self, _: S::Span) -> Self::Next {
        match self.parent.state {
            TokenContext::MacroDef(state) => {
                if let Some((name, params)) = state.label {
                    let tokens = state.tokens;
                    let id = self.parent.session.define_macro(name.1, params, tokens);
                    self.parent.session.insert(name.0, ResolvedName::Macro(id));
                }
            }
        }
        TokenStreamSemantics::new(self.parent.session)
    }
}
