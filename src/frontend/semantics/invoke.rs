use super::SemanticActions;
use crate::backend::Backend;
use crate::diag::{DelegateDiagnostics, Diagnostics};
use crate::frontend::syntax::{MacroInvocationContext, TokenSeqContext};
use crate::frontend::{Frontend, Ident, SemanticToken, TokenSeq};

pub(crate) struct MacroInvocationActions<'a, F: Frontend<D>, B: ?Sized, N, D: Diagnostics> {
    name: (Ident<F::StringRef>, D::Span),
    args: Vec<TokenSeq<F::StringRef, D::Span>>,
    parent: SemanticActions<'a, F, B, N, D>,
}

impl<'a, F: Frontend<D>, B: ?Sized, N, D: Diagnostics> MacroInvocationActions<'a, F, B, N, D> {
    pub fn new(
        name: (Ident<F::StringRef>, D::Span),
        parent: SemanticActions<'a, F, B, N, D>,
    ) -> MacroInvocationActions<'a, F, B, N, D> {
        MacroInvocationActions {
            name,
            args: Vec::new(),
            parent,
        }
    }

    fn push_arg(&mut self, arg: Vec<(SemanticToken<F::StringRef>, D::Span)>) {
        self.args.push(arg)
    }
}

impl<'a, F, B, N, D> DelegateDiagnostics<D::Span> for MacroInvocationActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.diagnostics()
    }
}

impl<'a, F, B, N, D> MacroInvocationContext<D::Span> for MacroInvocationActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    D: Diagnostics,
{
    type Token = SemanticToken<F::StringRef>;
    type Parent = SemanticActions<'a, F, B, N, D>;
    type MacroArgContext = MacroArgContext<'a, F, B, N, D>;

    fn enter_macro_arg(self) -> Self::MacroArgContext {
        MacroArgContext::new(self)
    }

    fn exit(mut self) -> Self::Parent {
        self.parent.session.invoke_macro(self.name, self.args);
        self.parent
    }
}

pub(crate) struct MacroArgContext<'a, F: Frontend<D>, B: ?Sized, N, D: Diagnostics> {
    tokens: Vec<(SemanticToken<F::StringRef>, D::Span)>,
    parent: MacroInvocationActions<'a, F, B, N, D>,
}

impl<'a, F: Frontend<D>, B: ?Sized, N, D: Diagnostics> MacroArgContext<'a, F, B, N, D> {
    fn new(parent: MacroInvocationActions<'a, F, B, N, D>) -> MacroArgContext<'a, F, B, N, D> {
        MacroArgContext {
            tokens: Vec::new(),
            parent,
        }
    }
}

impl<'a, F, B, N, D> DelegateDiagnostics<D::Span> for MacroArgContext<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.parent.diagnostics()
    }
}

impl<'a, F, B, N, D> TokenSeqContext<D::Span> for MacroArgContext<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type Token = SemanticToken<F::StringRef>;
    type Parent = MacroInvocationActions<'a, F, B, N, D>;

    fn push_token(&mut self, token: (Self::Token, D::Span)) {
        self.tokens.push(token)
    }

    fn exit(mut self) -> Self::Parent {
        self.parent.push_arg(self.tokens);
        self.parent
    }
}
