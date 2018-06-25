use std::{
    borrow::{Borrow, BorrowMut}, marker,
};
use {backend, diagnostics, frontend};

pub trait Session {
    type TokenRef: diagnostics::SourceInterval;
    fn analyze_chunk(&mut self, chunk_id: ChunkId<Self::TokenRef>);
    fn emit_diagnostic(&mut self, diagnostic: diagnostics::Diagnostic<Self::TokenRef>);
    fn emit_item(&mut self, item: backend::Item<Self::TokenRef>);
    fn define_label(&mut self, label: (String, Self::TokenRef));
    fn define_macro(
        &mut self,
        name: (impl Into<String>, Self::TokenRef),
        tokens: Vec<(frontend::Token, Self::TokenRef)>,
    );
    fn set_origin(&mut self, origin: backend::RelocExpr<Self::TokenRef>);
}

#[derive(Debug, PartialEq)]
pub enum ChunkId<T> {
    File((String, Option<T>)),
    Macro {
        name: (String, T),
        args: Vec<Vec<frontend::Token>>,
    },
}

pub struct Components<F, B, D, BMF, BMB, BD>
where
    F: frontend::Frontend,
    B: backend::Backend<F::TokenRef>,
    D: diagnostics::DiagnosticsListener<F::TokenRef>,
    BMF: BorrowMut<F>,
    BMB: BorrowMut<B>,
    BD: Borrow<D>,
{
    frontend: BMF,
    backend: BMB,
    diagnostics: BD,
    phantom: marker::PhantomData<(F, B, D)>,
}

pub type BorrowedComponents<'a, F, B, D> = Components<F, B, D, &'a mut F, &'a mut B, &'a D>;

impl<F, B, D, BMF, BMB, BD> Components<F, B, D, BMF, BMB, BD>
where
    F: frontend::Frontend,
    B: backend::Backend<F::TokenRef>,
    D: diagnostics::DiagnosticsListener<F::TokenRef>,
    BMF: BorrowMut<F>,
    BMB: BorrowMut<B>,
    BD: Borrow<D>,
{
    pub fn new(frontend: BMF, backend: BMB, diagnostics: BD) -> Components<F, B, D, BMF, BMB, BD> {
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

impl<F, B, D, BMF, BMB, BD> Session for Components<F, B, D, BMF, BMB, BD>
where
    F: frontend::Frontend,
    B: backend::Backend<F::TokenRef>,
    D: diagnostics::DiagnosticsListener<F::TokenRef>,
    BMF: BorrowMut<F>,
    BMB: BorrowMut<B>,
    BD: Borrow<D>,
{
    type TokenRef = F::TokenRef;

    fn analyze_chunk(&mut self, chunk_id: ChunkId<Self::TokenRef>) {
        self.frontend
            .borrow_mut()
            .analyze_chunk(chunk_id, self.backend.borrow_mut())
    }

    fn emit_diagnostic(&mut self, diagnostic: diagnostics::Diagnostic<Self::TokenRef>) {
        self.diagnostics.borrow().emit_diagnostic(diagnostic)
    }

    fn emit_item(&mut self, item: backend::Item<Self::TokenRef>) {
        self.backend.borrow_mut().emit_item(item)
    }

    fn define_label(&mut self, label: (String, Self::TokenRef)) {
        self.backend.borrow_mut().add_label(label)
    }

    fn define_macro(
        &mut self,
        name: (impl Into<String>, Self::TokenRef),
        tokens: Vec<(frontend::Token, Self::TokenRef)>,
    ) {
        self.frontend.borrow_mut().define_macro(name, tokens)
    }

    fn set_origin(&mut self, origin: backend::RelocExpr<Self::TokenRef>) {
        self.backend.borrow_mut().set_origin(origin)
    }
}
