use crate::backend;
use crate::codebase::CodebaseError;
use crate::diagnostics;
use crate::diagnostics::{DiagnosticsListener, InternalDiagnostic};
use crate::frontend;
use crate::frontend::{Downstream, Token};
use crate::span::{Merge, Span};

pub type MacroArgs<I, S> = Vec<Vec<(Token<I>, S)>>;

pub struct Components<'a, F, B, D> {
    pub frontend: &'a mut F,
    pub backend: &'a mut B,
    pub diagnostics: &'a mut D,
}

impl<'a, F, B, D> Components<'a, F, B, D> {
    pub fn new(
        frontend: &'a mut F,
        backend: &'a mut B,
        diagnostics: &'a mut D,
    ) -> Components<'a, F, B, D> {
        Components {
            frontend,
            backend,
            diagnostics,
        }
    }
}

impl<'a, F, B, D> Components<'a, F, B, D>
where
    F: frontend::Frontend<D>,
    D: diagnostics::Diagnostics,
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

impl<'a, F, B, D> Components<'a, F, B, D>
where
    F: frontend::Frontend<D>,
    B: backend::Backend<D::Span>,
    D: diagnostics::Diagnostics,
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

impl<'a, F, B, D> Span for Components<'a, F, B, D>
where
    F: frontend::Frontend<D>,
    B: backend::Backend<D::Span>,
    D: diagnostics::Diagnostics,
{
    type Span = D::Span;
}

impl<'a, F, B, D> Merge for Components<'a, F, B, D>
where
    F: frontend::Frontend<D>,
    B: backend::Backend<D::Span>,
    D: diagnostics::Diagnostics,
{
    fn merge(&mut self, left: &Self::Span, right: &Self::Span) -> Self::Span {
        self.diagnostics.diagnostics().merge(left, right)
    }
}

impl<'a, F, B, D> DiagnosticsListener for Components<'a, F, B, D>
where
    F: frontend::Frontend<D>,
    B: backend::Backend<D::Span>,
    D: diagnostics::Diagnostics,
{
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<Self::Span>) {
        self.diagnostics.diagnostics().emit_diagnostic(diagnostic)
    }
}

impl<'a, F, B, D> Components<'a, F, B, D>
where
    F: frontend::Frontend<D>,
    B: backend::Backend<D::Span>,
    D: diagnostics::Diagnostics,
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
