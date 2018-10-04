use backend;
use diagnostics;
use diagnostics::InternalDiagnostic;
use frontend;
use span;
use std::borrow::BorrowMut;
use std::marker;

pub trait Session {
    type Ident: Into<String>;
    type Span: span::Span;
    fn analyze_chunk(&mut self, chunk_id: ChunkId<Self::Ident, Self::Span>);
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<Self::Span>);
    fn emit_item(&mut self, item: backend::Item<Self::Span>);
    fn define_label(&mut self, label: (String, Self::Span));
    fn define_macro(
        &mut self,
        name: (impl Into<Self::Ident>, Self::Span),
        params: Vec<(Self::Ident, Self::Span)>,
        tokens: Vec<(frontend::Token<Self::Ident>, Self::Span)>,
    );
    fn set_origin(&mut self, origin: backend::RelocExpr<Self::Span>);
}

#[derive(Debug, PartialEq)]
pub enum ChunkId<I, S> {
    File((I, Option<S>)),
    Macro {
        name: (I, S),
        args: Vec<Vec<(frontend::Token<I>, S)>>,
    },
}

pub struct Components<F, B, D, BMF, BMB, BMD>
where
    F: frontend::Frontend,
    B: backend::Backend<F::Span>,
    D: diagnostics::DiagnosticsListener<F::Span>,
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
    F: frontend::Frontend,
    B: backend::Backend<F::Span>,
    D: diagnostics::DiagnosticsListener<F::Span>,
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

impl<F, B, D, BMF, BMB, BMD> Session for Components<F, B, D, BMF, BMB, BMD>
where
    F: frontend::Frontend,
    B: backend::Backend<F::Span>,
    D: diagnostics::DiagnosticsListener<F::Span>,
    BMF: BorrowMut<F>,
    BMB: BorrowMut<B>,
    BMD: BorrowMut<D>,
{
    type Ident = F::Ident;
    type Span = F::Span;

    fn analyze_chunk(&mut self, chunk_id: ChunkId<Self::Ident, Self::Span>) {
        self.frontend.borrow_mut().analyze_chunk(
            chunk_id,
            self.backend.borrow_mut(),
            self.diagnostics.borrow_mut(),
        )
    }

    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<Self::Span>) {
        self.diagnostics.borrow_mut().emit_diagnostic(diagnostic)
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
        tokens: Vec<(frontend::Token<Self::Ident>, Self::Span)>,
    ) {
        self.frontend
            .borrow_mut()
            .define_macro(name, params, tokens)
    }

    fn set_origin(&mut self, origin: backend::RelocExpr<Self::Span>) {
        self.backend.borrow_mut().set_origin(origin)
    }
}
