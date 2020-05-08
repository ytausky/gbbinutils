use super::{Keyword, TokenStreamSemantics};

use crate::analyze::reentrancy::ReentrancyActions;
use crate::analyze::semantics::keywords::{FreeBuiltinMnemonic, FreeDirective};
use crate::analyze::semantics::resolve::{NameTable, ResolvedName, StartScope};
use crate::analyze::semantics::*;
use crate::analyze::syntax::actions::{LineFinalizer, TokenLineActions, TokenLineRule};
use crate::analyze::syntax::{Sigil, Token};
use crate::analyze::{Literal, SemanticToken};

use std::ops::DerefMut;

impl<R, N, B> TokenLineActions<R::Ident, Literal<R::StringRef>, R::Span>
    for TokenLineSemantics<R, N, B>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<R::Ident, Keyword = &'static Keyword, MacroId = R::MacroId>,
{
    type ContextFinalizer = TokenContextFinalizationSemantics<R, N, B>;

    fn act_on_token(&mut self, token: SemanticToken<R::Ident, R::StringRef>, span: R::Span) {
        match &mut self.state.context {
            TokenLineContext::FalseIf => (),
            TokenLineContext::MacroDef(state) => state.act_on_token(token, span),
        }
    }

    fn act_on_mnemonic(
        mut self,
        ident: R::Ident,
        span: R::Span,
    ) -> TokenLineRule<Self, Self::ContextFinalizer> {
        if let Some(ResolvedName::Keyword(Keyword::BuiltinMnemonic(mnemonic))) =
            self.resolve_name(&ident)
        {
            if let TokenLineRule::LineEnd(()) =
                self.state.context.act_on_mnemonic(mnemonic, span.clone())
            {
                return TokenLineRule::LineEnd(TokenContextFinalizationSemantics { parent: self });
            }
        }
        self.act_on_token(Token::Ident(ident), span);
        TokenLineRule::TokenSeq(self)
    }
}

pub(in crate::analyze) trait TokenContext<I, R: ReentrancyActions>:
    ActOnMnemonic<&'static BuiltinMnemonic, R::Span>
    + ActOnToken<SemanticToken<R::Ident, R::StringRef>, R::Span>
{
}

impl<T, I, R> TokenContext<I, R> for T
where
    T: ActOnMnemonic<&'static BuiltinMnemonic, R::Span>
        + ActOnToken<SemanticToken<R::Ident, R::StringRef>, R::Span>,
    R: ReentrancyActions,
{
}

pub(in crate::analyze) trait ActOnMnemonic<M, S> {
    fn act_on_mnemonic(&mut self, mnemonic: M, span: S) -> TokenLineRule<(), ()>;
}

impl<I, R, S> ActOnMnemonic<&'static BuiltinMnemonic, S> for TokenLineContext<I, R, S>
where
    Self: ActOnToken<SemanticToken<I, R>, S>,
{
    fn act_on_mnemonic(
        &mut self,
        mnemonic: &'static BuiltinMnemonic,
        span: S,
    ) -> TokenLineRule<(), ()> {
        match (&*self, mnemonic) {
            (
                TokenLineContext::FalseIf,
                BuiltinMnemonic::Free(FreeBuiltinMnemonic::Directive(FreeDirective::Endc)),
            ) => TokenLineRule::LineEnd(()),
            (
                TokenLineContext::MacroDef(_),
                BuiltinMnemonic::Free(FreeBuiltinMnemonic::Directive(FreeDirective::Endm)),
            ) => {
                self.act_on_token(Sigil::Eos.into(), span);
                TokenLineRule::LineEnd(())
            }
            _ => TokenLineRule::TokenSeq(()),
        }
    }
}

pub(in crate::analyze) trait ActOnToken<T, S> {
    fn act_on_token(&mut self, token: T, span: S);
}

impl<I, R, S> ActOnToken<SemanticToken<I, R>, S> for TokenLineContext<I, R, S> {
    fn act_on_token(&mut self, token: SemanticToken<I, R>, span: S) {
        match self {
            TokenLineContext::FalseIf => drop((token, span)),
            TokenLineContext::MacroDef(state) => state.act_on_token(token, span),
        }
    }
}

impl<I, R, S> MacroDefState<I, R, S> {
    fn act_on_token(&mut self, token: SemanticToken<I, R>, span: S) {
        self.tokens.0.push(token);
        self.tokens.1.push(span);
    }
}

impl<R, N, B> LineFinalizer<R::Span> for TokenLineSemantics<R, N, B>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<R::Ident, Keyword = &'static Keyword, MacroId = R::MacroId>,
{
    type Next = TokenStreamSemantics<R, N, B>;

    fn did_parse_line(mut self, span: R::Span) -> Self::Next {
        self.act_on_token(Sigil::Eol.into(), span);
        set_state!(self, self.state.into())
    }
}

pub(in crate::analyze) struct TokenContextFinalizationSemantics<R: ReentrancyActions, N, B> {
    parent: TokenLineSemantics<R, N, B>,
}

delegate_diagnostics! {
    {R: ReentrancyActions, N, B},
    TokenContextFinalizationSemantics<R, N, B>,
    {parent},
    R,
    R::Span
}

impl<R, N, B> LineFinalizer<R::Span> for TokenContextFinalizationSemantics<R, N, B>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: NameTable<R::Ident, MacroId = R::MacroId>,
{
    type Next = TokenStreamSemantics<R, N, B>;

    fn did_parse_line(mut self, _: R::Span) -> Self::Next {
        match self.parent.state.context {
            TokenLineContext::FalseIf => (),
            TokenLineContext::MacroDef(state) => {
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
