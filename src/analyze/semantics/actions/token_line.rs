use super::{Keyword, TokenStreamSemantics};

use crate::analyze::semantics::keywords::Directive;
use crate::analyze::semantics::reentrancy::ReentrancyActions;
use crate::analyze::semantics::resolve::{NameTable, ResolvedName, StartScope};
use crate::analyze::semantics::*;
use crate::analyze::syntax::actions::*;
use crate::analyze::syntax::{LexError, Sigil, Token};
use crate::analyze::{Literal, SemanticToken};
use crate::diag::span::StripSpan;
use crate::diag::CompactDiag;

use std::ops::DerefMut;

impl<'a, R, N, B> TokenLineContext for TokenLineSemantics<'a, R, N, B>
where
    R: Meta,
    Core<R, N, B>: ReentrancyActions<
        Ident = R::Ident,
        StringRef = R::StringRef,
        Span = R::Span,
        MacroId = R::MacroId,
    >,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<R::Ident, Keyword = &'static Keyword, MacroId = R::MacroId>,
{
    type ContextFinalizer = TokenContextFinalizationSemantics<'a, R, N, B>;

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

impl<'a, R, N, B> LineFinalizer for TokenLineSemantics<'a, R, N, B>
where
    R: Meta,
    Core<R, N, B>: ReentrancyActions<
        Ident = R::Ident,
        StringRef = R::StringRef,
        Span = R::Span,
        MacroId = R::MacroId,
    >,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<R::Ident, Keyword = &'static Keyword, MacroId = R::MacroId>,
{
    type Next = TokenStreamSemantics<'a, R, N, B>;

    fn did_parse_line(mut self, span: R::Span) -> Self::Next {
        self.act_on_token(Sigil::Eol.into(), span);
        set_state!(self, self.state.into())
    }
}

pub(in crate::analyze) struct TokenContextFinalizationSemantics<'a, R: Meta, N, B> {
    parent: TokenLineSemantics<'a, R, N, B>,
}

impl<'a, R: Meta, N, B> ParsingContext for TokenContextFinalizationSemantics<'a, R, N, B> {
    type Ident = R::Ident;
    type Literal = Literal<R::StringRef>;
    type Error = LexError;
    type Span = R::Span;
    type Stripped = <R as StripSpan<R::Span>>::Stripped;

    fn next_token(
        &mut self,
    ) -> Option<LexerOutput<Self::Ident, Self::Literal, Self::Error, Self::Span>> {
        self.parent.tokens.next()
    }

    fn merge_spans(&mut self, left: &Self::Span, right: &Self::Span) -> Self::Span {
        self.parent.core.reentrancy.merge_spans(left, right)
    }

    fn strip_span(&mut self, span: &Self::Span) -> Self::Stripped {
        self.parent.core.reentrancy.strip_span(span)
    }

    fn emit_diag(&mut self, diag: impl Into<CompactDiag<Self::Span, Self::Stripped>>) {
        self.parent.core.reentrancy.emit_diag(diag)
    }
}

impl<'a, R, N, B> LineFinalizer for TokenContextFinalizationSemantics<'a, R, N, B>
where
    R: Meta,
    Core<R, N, B>: ReentrancyActions<
        Ident = R::Ident,
        StringRef = R::StringRef,
        Span = R::Span,
        MacroId = R::MacroId,
    >,
    N: DerefMut,
    N::Target: NameTable<R::Ident, MacroId = R::MacroId>,
{
    type Next = TokenStreamSemantics<'a, R, N, B>;

    fn did_parse_line(mut self, _: R::Span) -> Self::Next {
        match self.parent.state.context {
            TokenContext::FalseIf => (),
            TokenContext::MacroDef(state) => {
                if let Some((name, params)) = state.label {
                    let tokens = state.tokens;
                    let id = self.parent.core.define_macro(name.1, params, tokens);
                    self.parent
                        .core
                        .names
                        .define_name(name.0, ResolvedName::Macro(id));
                }
            }
        }
        set_state!(self.parent, TokenStreamState::new())
    }
}
