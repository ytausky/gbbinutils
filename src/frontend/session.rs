use crate::backend;
use crate::codebase::CodebaseError;
use crate::diagnostics;
use crate::diagnostics::{DiagnosticsListener, InternalDiagnostic};
use crate::frontend;
use crate::frontend::{Downstream, Token};
use crate::span::{Merge, Span};
use std::fmt::Debug;

pub trait Session
where
    Self: Merge,
    Self: DiagnosticsListener,
{
    type Ident: Into<String> + Debug + PartialEq;
    fn analyze_file(&mut self, path: Self::Ident) -> Result<(), CodebaseError>;
    fn invoke_macro(
        &mut self,
        name: (Self::Ident, Self::Span),
        args: MacroArgs<Self::Ident, Self::Span>,
    );
    fn emit_item(&mut self, item: backend::Item<Self::Span>);
    fn define_label(&mut self, label: (String, Self::Span));
    fn define_macro(
        &mut self,
        name: (impl Into<Self::Ident>, Self::Span),
        params: Vec<(Self::Ident, Self::Span)>,
        tokens: Vec<(Token<Self::Ident>, Self::Span)>,
    );
    fn set_origin(&mut self, origin: backend::RelocExpr<Self::Span>);
}

pub type MacroArgs<I, S> = Vec<Vec<(Token<I>, S)>>;

pub struct Components<'a, F, B, D>
where
    F: frontend::Frontend<D>,
    B: backend::Backend<D::Span>,
    D: diagnostics::Diagnostics,
{
    frontend: &'a mut F,
    backend: &'a mut B,
    diagnostics: &'a mut D,
}

impl<'a, F, B, D> Components<'a, F, B, D>
where
    F: frontend::Frontend<D>,
    B: backend::Backend<D::Span>,
    D: diagnostics::Diagnostics,
{
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

impl<'a, F, B, D> Session for Components<'a, F, B, D>
where
    F: frontend::Frontend<D>,
    B: backend::Backend<D::Span>,
    D: diagnostics::Diagnostics,
{
    type Ident = F::Ident;

    fn analyze_file(&mut self, path: Self::Ident) -> Result<(), CodebaseError> {
        self.frontend.analyze_file(
            path,
            Downstream {
                backend: self.backend,
                diagnostics: self.diagnostics,
            },
        )
    }

    fn invoke_macro(
        &mut self,
        name: (Self::Ident, Self::Span),
        args: MacroArgs<Self::Ident, Self::Span>,
    ) {
        self.frontend.invoke_macro(
            name,
            args,
            Downstream {
                backend: self.backend,
                diagnostics: self.diagnostics,
            },
        )
    }

    fn emit_item(&mut self, item: backend::Item<Self::Span>) {
        self.backend.emit_item(item)
    }

    fn define_label(&mut self, label: (String, Self::Span)) {
        self.backend.add_label(label)
    }

    fn define_macro(
        &mut self,
        name: (impl Into<Self::Ident>, Self::Span),
        params: Vec<(Self::Ident, Self::Span)>,
        tokens: Vec<(Token<Self::Ident>, Self::Span)>,
    ) {
        self.frontend
            .define_macro(name, params, tokens, &mut self.diagnostics)
    }

    fn set_origin(&mut self, origin: backend::RelocExpr<Self::Span>) {
        self.backend.set_origin(origin)
    }
}
