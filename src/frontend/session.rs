use crate::backend::Backend;
use crate::codebase::CodebaseError;
use crate::diag::{DelegateDiagnostics, Diagnostics, DownstreamDiagnostics};
use crate::frontend::{Downstream, Frontend, Token};

pub type MacroArgs<I, S> = Vec<Vec<(Token<I>, S)>>;

pub(crate) struct Session<'a, F, B, D> {
    pub frontend: &'a mut F,
    pub backend: &'a mut B,
    pub diagnostics: &'a mut D,
}

impl<'a, F, B, D> Session<'a, F, B, D> {
    pub fn new(
        frontend: &'a mut F,
        backend: &'a mut B,
        diagnostics: &'a mut D,
    ) -> Session<'a, F, B, D> {
        Session {
            frontend,
            backend,
            diagnostics,
        }
    }
}

impl<'a, F, B, D> Session<'a, F, B, D>
where
    F: Frontend<D>,
    D: Diagnostics,
{
    pub fn define_macro(
        &mut self,
        name: (impl Into<F::Ident>, D::Span),
        params: Vec<(F::Ident, D::Span)>,
        tokens: Vec<(Token<F::Ident>, D::Span)>,
    ) {
        self.frontend
            .define_macro(name, params, tokens, &mut self.diagnostics)
    }
}

impl<'a, F, B, D> Session<'a, F, B, D>
where
    F: Frontend<D>,
    B: Backend<F::Ident, D::Span>,
    D: Diagnostics,
{
    pub fn analyze_file(&mut self, path: F::Ident) -> Result<(), CodebaseError> {
        self.frontend.analyze_file(
            path,
            Downstream {
                backend: self.backend,
                diagnostics: self.diagnostics,
            },
        )
    }
}

impl<'a, F, B, D: DownstreamDiagnostics<S>, S> DelegateDiagnostics<S> for Session<'a, F, B, D> {
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.diagnostics
    }
}

impl<'a, F, B, D> Session<'a, F, B, D>
where
    F: Frontend<D>,
    B: Backend<F::Ident, D::Span>,
    D: Diagnostics,
{
    pub fn invoke_macro(&mut self, name: (F::Ident, D::Span), args: MacroArgs<F::Ident, D::Span>) {
        self.frontend.invoke_macro(
            name,
            args,
            Downstream {
                backend: self.backend,
                diagnostics: self.diagnostics,
            },
        )
    }
}
