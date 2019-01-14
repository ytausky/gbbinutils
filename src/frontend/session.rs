use crate::backend::Backend;
use crate::codebase::CodebaseError;
use crate::diag::{DelegateDiagnostics, Diagnostics, DownstreamDiagnostics};
use crate::frontend::{Downstream, Frontend, Ident, SemanticToken};

pub(super) type MacroArgs<I, S> = Vec<Vec<(SemanticToken<I>, S)>>;

pub(crate) struct Session<'a, F, B: ?Sized, N, D> {
    pub frontend: &'a mut F,
    pub backend: &'a mut B,
    pub names: &'a mut N,
    pub diagnostics: &'a mut D,
}

impl<'a, F, B: ?Sized, N, D> Session<'a, F, B, N, D> {
    pub fn new(
        frontend: &'a mut F,
        backend: &'a mut B,
        names: &'a mut N,
        diagnostics: &'a mut D,
    ) -> Session<'a, F, B, N, D> {
        Session {
            frontend,
            backend,
            names,
            diagnostics,
        }
    }
}

impl<'a, F, B, N, D> Session<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    D: Diagnostics,
{
    pub fn analyze_file(&mut self, path: F::StringRef) -> Result<(), CodebaseError> {
        self.frontend.analyze_file(
            path,
            Downstream {
                backend: self.backend,
                names: self.names,
                diagnostics: self.diagnostics,
            },
        )
    }
}

impl<'a, F, B, N, D: DownstreamDiagnostics<S>, S> DelegateDiagnostics<S>
    for Session<'a, F, B, N, D>
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.diagnostics
    }
}

impl<'a, F, B, N, D> Session<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    D: Diagnostics,
{
    pub fn invoke_macro(
        &mut self,
        name: (Ident<F::StringRef>, D::Span),
        args: MacroArgs<F::StringRef, D::Span>,
    ) {
        self.frontend.invoke_macro(
            name,
            args,
            Downstream {
                backend: self.backend,
                names: self.names,
                diagnostics: self.diagnostics,
            },
        )
    }
}
