use super::{Keyword, TokenStreamSemantics};

use crate::analyze::semantics::keywords::Directive;
use crate::analyze::semantics::*;
use crate::analyze::{Literal, SemanticToken};
use crate::diag::span::StripSpan;
use crate::diag::CompactDiag;
use crate::session::resolve::ResolvedName;
use crate::syntax::actions::*;
use crate::syntax::{LexError, Sigil, Token};

impl<'a, S: Session> TokenLineContext for TokenLineSemantics<'a, S> {
    type ContextFinalizer = TokenContextFinalizationSemantics<'a, S>;

    fn act_on_token(&mut self, token: SemanticToken<S::Ident, S::StringRef>, span: S::Span) {
        match &mut self.state.context {
            TokenContext::FalseIf => (),
            TokenContext::MacroDef(state) => state.act_on_token(token, span),
        }
    }

    fn act_on_mnemonic(
        mut self,
        ident: S::Ident,
        span: S::Span,
    ) -> TokenLineRule<Self, Self::ContextFinalizer> {
        if let Some(ResolvedName::Keyword(Keyword::BuiltinMnemonic(mnemonic))) =
            self.session.resolve_name(&ident)
        {
            if let TokenLineRule::LineEnd(()) =
                self.state.context.act_on_mnemonic(&mnemonic, span.clone())
            {
                return TokenLineRule::LineEnd(TokenContextFinalizationSemantics { parent: self });
            }
        }
        self.act_on_token(Token::Ident(ident), span);
        TokenLineRule::TokenSeq(self)
    }
}

pub(in crate::analyze) trait ActOnMnemonic<M, S> {
    fn act_on_mnemonic(&mut self, mnemonic: M, span: S) -> TokenLineRule<(), ()>;
}

impl<I, R, S> ActOnMnemonic<&'static BuiltinMnemonic, S> for TokenContext<I, R, S>
where
    Self: ActOnToken<SemanticToken<I, R>, S>,
{
    fn act_on_mnemonic(
        &mut self,
        mnemonic: &'static BuiltinMnemonic,
        span: S,
    ) -> TokenLineRule<(), ()> {
        match (&*self, mnemonic) {
            (TokenContext::FalseIf, BuiltinMnemonic::Directive(Directive::Endc)) => {
                TokenLineRule::LineEnd(())
            }
            (TokenContext::MacroDef(_), BuiltinMnemonic::Directive(Directive::Endm)) => {
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

impl<I, R, S> ActOnToken<SemanticToken<I, R>, S> for TokenContext<I, R, S> {
    fn act_on_token(&mut self, token: SemanticToken<I, R>, span: S) {
        match self {
            TokenContext::FalseIf => drop((token, span)),
            TokenContext::MacroDef(state) => state.act_on_token(token, span),
        }
    }
}

impl<I, R, S> MacroDefState<I, R, S> {
    fn act_on_token(&mut self, token: SemanticToken<I, R>, span: S) {
        self.tokens.0.push(token);
        self.tokens.1.push(span);
    }
}

impl<'a, S: Session> LineFinalizer for TokenLineSemantics<'a, S> {
    type Next = TokenStreamSemantics<'a, S>;

    fn did_parse_line(mut self, span: S::Span) -> Self::Next {
        self.act_on_token(Sigil::Eol.into(), span);
        set_state!(self, self.state.into())
    }
}

pub(crate) struct TokenContextFinalizationSemantics<'a, S: Session> {
    parent: TokenLineSemantics<'a, S>,
}

impl<'a, S: Session> ParsingContext for TokenContextFinalizationSemantics<'a, S> {
    type Ident = S::Ident;
    type Literal = Literal<S::StringRef>;
    type Error = LexError;
    type Span = S::Span;
    type Stripped = <S as StripSpan<S::Span>>::Stripped;

    fn next_token(
        &mut self,
    ) -> Option<LexerOutput<Self::Ident, Self::Literal, Self::Error, Self::Span>> {
        self.parent.tokens.next()
    }

    fn merge_spans(&mut self, left: &Self::Span, right: &Self::Span) -> Self::Span {
        self.parent.session.merge_spans(left, right)
    }

    fn strip_span(&mut self, span: &Self::Span) -> Self::Stripped {
        self.parent.session.strip_span(span)
    }

    fn emit_diag(&mut self, diag: impl Into<CompactDiag<Self::Span, Self::Stripped>>) {
        self.parent.session.emit_diag(diag)
    }
}

impl<'a, S: Session> LineFinalizer for TokenContextFinalizationSemantics<'a, S> {
    type Next = TokenStreamSemantics<'a, S>;

    fn did_parse_line(mut self, _: S::Span) -> Self::Next {
        match self.parent.state.context {
            TokenContext::FalseIf => (),
            TokenContext::MacroDef(state) => {
                if let Some((name, params)) = state.label {
                    let tokens = state.tokens;
                    let id = self.parent.session.define_macro(name.1, params, tokens);
                    self.parent
                        .session
                        .define_name(name.0, ResolvedName::Macro(id));
                }
            }
        }
        set_state!(self.parent, TokenStreamState::new())
    }
}
