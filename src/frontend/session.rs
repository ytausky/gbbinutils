use crate::backend;
use crate::backend::{
    ApplyBinaryOperator, Backend, HasValue, Item, NameTable, PartialBackend, ValueFromSimple,
};
use crate::codebase::CodebaseError;
use crate::diag::span::Span;
use crate::diag::{DelegateDiagnostics, Diagnostics, DownstreamDiagnostics};
use crate::expr::BinaryOperator;
use crate::frontend::macros::MacroEntry;
use crate::frontend::{Downstream, Frontend, Ident, SemanticToken, StringRef};

#[cfg(test)]
pub use self::mock::*;

pub(crate) trait Session
where
    Self: Span + StringRef,
    Self: DelegateDiagnostics<<Self as Span>::Span>,
    Self: PartialBackend<<Self as Span>::Span>,
    Self: ValueBuilder<Ident<<Self as StringRef>::StringRef>, <Self as Span>::Span>,
{
    fn analyze_file(&mut self, path: Self::StringRef) -> Result<(), CodebaseError>;
    fn define_macro(
        &mut self,
        name: (Ident<Self::StringRef>, Self::Span),
        params: (Vec<Ident<Self::StringRef>>, Vec<Self::Span>),
        body: (Vec<SemanticToken<Self::StringRef>>, Vec<Self::Span>),
    );
    fn invoke_macro(
        &mut self,
        name: (Ident<Self::StringRef>, Self::Span),
        args: MacroArgs<Self::StringRef, Self::Span>,
    );
    fn define_symbol(&mut self, symbol: (Ident<Self::StringRef>, Self::Span), value: Self::Value);
}

pub(crate) trait ValueBuilder<I, S: Clone>
where
    Self: HasValue<S>,
    Self: backend::ValueFromSimple<S>,
    Self: backend::ApplyBinaryOperator<S>,
{
    fn from_ident(&mut self, ident: I, span: S) -> Self::Value;
}

pub(super) type MacroArgs<I, S> = Vec<Vec<(SemanticToken<I>, S)>>;

pub(crate) struct CompositeSession<'a, F, B: ?Sized, N, D> {
    frontend: &'a mut F,
    backend: &'a mut B,
    names: &'a mut N,
    diagnostics: &'a mut D,
}

impl<'a, F, B: ?Sized, N, D> CompositeSession<'a, F, B, N, D> {
    pub fn new(
        frontend: &'a mut F,
        backend: &'a mut B,
        names: &'a mut N,
        diagnostics: &'a mut D,
    ) -> CompositeSession<'a, F, B, N, D> {
        CompositeSession {
            frontend,
            backend,
            names,
            diagnostics,
        }
    }
}

macro_rules! downstream {
    ($session:expr) => {
        Downstream {
            backend: $session.backend,
            names: $session.names,
            diagnostics: $session.diagnostics,
        }
    };
}

impl<'a, F, B, N, D> Span for CompositeSession<'a, F, B, N, D>
where
    B: ?Sized,
    D: Span,
{
    type Span = D::Span;
}

impl<'a, F, B, N, D> StringRef for CompositeSession<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type StringRef = F::StringRef;
}

impl<'a, F, B, N, D> HasValue<D::Span> for CompositeSession<'a, F, B, N, D>
where
    B: HasValue<D::Span> + ?Sized,
    D: Span,
{
    type Value = B::Value;
}

impl<'a, F, B, N, D> PartialBackend<D::Span> for CompositeSession<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    N: NameTable<Ident<F::StringRef>, MacroEntry = MacroEntry<F, D>>,
    D: Diagnostics,
{
    fn emit_item(&mut self, item: Item<Self::Value>) {
        self.backend.emit_item(item)
    }

    fn set_origin(&mut self, origin: Self::Value) {
        self.backend.set_origin(origin)
    }
}

impl<'a, F, B, N, D> ValueFromSimple<D::Span> for CompositeSession<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    N: NameTable<Ident<F::StringRef>, MacroEntry = MacroEntry<F, D>>,
    D: Diagnostics,
{
    fn from_location_counter(&mut self, span: D::Span) -> Self::Value {
        self.backend.from_location_counter(span)
    }

    fn from_number(&mut self, n: i32, span: D::Span) -> Self::Value {
        self.backend.from_number(n, span)
    }
}

impl<'a, F, B, N, D> ApplyBinaryOperator<D::Span> for CompositeSession<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    N: NameTable<Ident<F::StringRef>, MacroEntry = MacroEntry<F, D>>,
    D: Diagnostics,
{
    fn apply_binary_operator(
        &mut self,
        operator: (BinaryOperator, D::Span),
        left: Self::Value,
        right: Self::Value,
    ) -> Self::Value {
        self.backend.apply_binary_operator(operator, left, right)
    }
}

impl<'a, F, B, N, D> ValueBuilder<Ident<F::StringRef>, D::Span> for CompositeSession<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    N: NameTable<Ident<F::StringRef>, MacroEntry = MacroEntry<F, D>>,
    D: Diagnostics,
{
    fn from_ident(&mut self, ident: Ident<F::StringRef>, span: D::Span) -> Self::Value {
        self.backend.from_ident(ident, span, self.names)
    }
}

impl<'a, F, B, N, D> Session for CompositeSession<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    N: NameTable<Ident<F::StringRef>, MacroEntry = MacroEntry<F, D>>,
    D: Diagnostics,
{
    fn analyze_file(&mut self, path: Self::StringRef) -> Result<(), CodebaseError> {
        self.frontend.analyze_file(path, downstream!(self))
    }

    fn define_macro(
        &mut self,
        name: (Ident<Self::StringRef>, Self::Span),
        params: (Vec<Ident<Self::StringRef>>, Vec<Self::Span>),
        body: (Vec<SemanticToken<Self::StringRef>>, Vec<Self::Span>),
    ) {
        self.frontend
            .define_macro(name, params, body, downstream!(self))
    }

    fn invoke_macro(
        &mut self,
        name: (Ident<Self::StringRef>, Self::Span),
        args: MacroArgs<Self::StringRef, Self::Span>,
    ) {
        self.frontend.invoke_macro(name, args, downstream!(self))
    }

    fn define_symbol(&mut self, symbol: (Ident<Self::StringRef>, Self::Span), value: Self::Value) {
        self.backend.define_symbol(symbol, value, &mut self.names)
    }
}

impl<'a, F, B, N, D> CompositeSession<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    N: NameTable<Ident<F::StringRef>, MacroEntry = MacroEntry<F, D>>,
    D: Diagnostics,
{
    pub fn analyze_file(&mut self, path: F::StringRef) -> Result<(), CodebaseError> {
        self.frontend.analyze_file(path, downstream!(self))
    }
}

impl<'a, F, B, N, D, S> DelegateDiagnostics<S> for CompositeSession<'a, F, B, N, D>
where
    B: ?Sized,
    D: DownstreamDiagnostics<S>,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.diagnostics
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::backend::{RelocAtom, RelocExpr};
    use crate::diag;
    use crate::diag::{MockDiagnostics, MockSpan};
    use crate::expr::{Expr, ExprVariant};
    use std::cell::RefCell;

    pub struct MockSession<'a, T, S> {
        _log: &'a RefCell<Vec<T>>,
        diagnostics: MockDiagnostics<'a, T, S>,
    }

    impl<'a, T, S> MockSession<'a, T, S> {
        pub fn new(log: &'a RefCell<Vec<T>>) -> Self {
            Self {
                _log: log,
                diagnostics: MockDiagnostics::new(log),
            }
        }
    }

    impl<'a, T, S: Clone> HasValue<S> for MockSession<'a, T, S> {
        type Value = RelocExpr<Ident<String>, S>;
    }

    impl<'a, T, S: Clone> ValueFromSimple<S> for MockSession<'a, T, S> {
        fn from_location_counter(&mut self, span: S) -> Self::Value {
            RelocExpr::from_atom(RelocAtom::LocationCounter, span)
        }

        fn from_number(&mut self, n: i32, span: S) -> Self::Value {
            RelocExpr::from_atom(RelocAtom::Literal(n), span)
        }
    }

    impl<'a, T, S: Clone> ApplyBinaryOperator<S> for MockSession<'a, T, S> {
        fn apply_binary_operator(
            &mut self,
            operator: (BinaryOperator, S),
            left: Self::Value,
            right: Self::Value,
        ) -> Self::Value {
            Expr {
                variant: ExprVariant::Binary(operator.0, Box::new(left), Box::new(right)),
                span: operator.1,
            }
        }
    }

    impl<'a, T, S: Clone> ValueBuilder<Ident<String>, S> for MockSession<'a, T, S> {
        fn from_ident(&mut self, ident: Ident<String>, span: S) -> Self::Value {
            RelocExpr::from_atom(RelocAtom::Symbol(ident), span)
        }
    }

    impl<'a, T, S> DelegateDiagnostics<S> for MockSession<'a, T, S>
    where
        T: From<diag::Event<S>>,
        S: Clone + MockSpan,
    {
        type Delegate = MockDiagnostics<'a, T, S>;

        fn diagnostics(&mut self) -> &mut Self::Delegate {
            &mut self.diagnostics
        }
    }
}
