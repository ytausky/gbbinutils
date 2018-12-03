use crate::backend;
use crate::codebase::CodebaseError;
use crate::diagnostics;
use crate::diagnostics::{DiagnosticsListener, InternalDiagnostic};
use crate::frontend;
use crate::frontend::{Downstream, Token};
use crate::span::{HasSpan, Merge};
use std::borrow::BorrowMut;
use std::fmt::Debug;
use std::marker;

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

pub struct Components<F, B, D, BMF, BMB, BMD>
where
    F: frontend::Frontend<D>,
    B: backend::Backend<D::Span>,
    D: diagnostics::Diagnostics,
    BMF: BorrowMut<F>,
    BMB: BorrowMut<B>,
    BMD: BorrowMut<D>,
{
    frontend: BMF,
    backend: BMB,
    diagnostics: BMD,
    phantom: marker::PhantomData<(F, B, D)>,
}

pub type BorrowedComponents<'a, F, B, D> = Components<F, B, D, &'a mut F, &'a mut B, &'a mut D>;

impl<F, B, D, BMF, BMB, BMD> Components<F, B, D, BMF, BMB, BMD>
where
    F: frontend::Frontend<D>,
    B: backend::Backend<D::Span>,
    D: diagnostics::Diagnostics,
    BMF: BorrowMut<F>,
    BMB: BorrowMut<B>,
    BMD: BorrowMut<D>,
{
    pub fn new(
        frontend: BMF,
        backend: BMB,
        diagnostics: BMD,
    ) -> Components<F, B, D, BMF, BMB, BMD> {
        Components {
            frontend,
            backend,
            diagnostics,
            phantom: marker::PhantomData,
        }
    }

    pub fn build_object(self) -> B::Object
    where
        BMB: Into<B>,
    {
        self.backend.into().into_object()
    }
}

impl<F, B, D, BMF, BMB, BMD> HasSpan for Components<F, B, D, BMF, BMB, BMD>
where
    F: frontend::Frontend<D>,
    B: backend::Backend<D::Span>,
    D: diagnostics::Diagnostics,
    BMF: BorrowMut<F>,
    BMB: BorrowMut<B>,
    BMD: BorrowMut<D>,
{
    type Span = D::Span;
}

impl<F, B, D, BMF, BMB, BMD> Merge for Components<F, B, D, BMF, BMB, BMD>
where
    F: frontend::Frontend<D>,
    B: backend::Backend<D::Span>,
    D: diagnostics::Diagnostics,
    BMF: BorrowMut<F>,
    BMB: BorrowMut<B>,
    BMD: BorrowMut<D>,
{
    fn merge(&mut self, left: &Self::Span, right: &Self::Span) -> Self::Span {
        self.diagnostics
            .borrow_mut()
            .diagnostics()
            .merge(left, right)
    }
}

impl<F, B, D, BMF, BMB, BMD> DiagnosticsListener for Components<F, B, D, BMF, BMB, BMD>
where
    F: frontend::Frontend<D>,
    B: backend::Backend<D::Span>,
    D: diagnostics::Diagnostics,
    BMF: BorrowMut<F>,
    BMB: BorrowMut<B>,
    BMD: BorrowMut<D>,
{
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<Self::Span>) {
        self.diagnostics
            .borrow_mut()
            .diagnostics()
            .emit_diagnostic(diagnostic)
    }
}

impl<F, B, D, BMF, BMB, BMD> Session for Components<F, B, D, BMF, BMB, BMD>
where
    F: frontend::Frontend<D>,
    B: backend::Backend<D::Span>,
    D: diagnostics::Diagnostics,
    BMF: BorrowMut<F>,
    BMB: BorrowMut<B>,
    BMD: BorrowMut<D>,
{
    type Ident = F::Ident;

    fn analyze_file(&mut self, path: Self::Ident) -> Result<(), CodebaseError> {
        self.frontend.borrow_mut().analyze_file(
            path,
            Downstream {
                backend: self.backend.borrow_mut(),
                diagnostics: self.diagnostics.borrow_mut(),
            },
        )
    }

    fn invoke_macro(
        &mut self,
        name: (Self::Ident, Self::Span),
        args: MacroArgs<Self::Ident, Self::Span>,
    ) {
        self.frontend.borrow_mut().invoke_macro(
            name,
            args,
            Downstream {
                backend: self.backend.borrow_mut(),
                diagnostics: self.diagnostics.borrow_mut(),
            },
        )
    }

    fn emit_item(&mut self, item: backend::Item<Self::Span>) {
        self.backend.borrow_mut().emit_item(item)
    }

    fn define_label(&mut self, label: (String, Self::Span)) {
        self.backend.borrow_mut().add_label(label)
    }

    fn define_macro(
        &mut self,
        name: (impl Into<Self::Ident>, Self::Span),
        params: Vec<(Self::Ident, Self::Span)>,
        tokens: Vec<(Token<Self::Ident>, Self::Span)>,
    ) {
        self.frontend
            .borrow_mut()
            .define_macro(name, params, tokens, self.diagnostics.borrow_mut())
    }

    fn set_origin(&mut self, origin: backend::RelocExpr<Self::Span>) {
        self.backend.borrow_mut().set_origin(origin)
    }
}
