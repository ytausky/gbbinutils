use crate::backend;
use crate::backend::{
    ApplyBinaryOperator, Backend, HasValue, Item, NameTable, PartialBackend, ValueFromSimple,
};
#[cfg(test)]
use crate::backend::{RelocAtom, RelocExpr};
use crate::codebase::CodebaseError;
use crate::diag::span::Span;
use crate::diag::{DelegateDiagnostics, Diagnostics, DownstreamDiagnostics};
use crate::expr::BinaryOperator;
#[cfg(test)]
use crate::expr::{Expr, ExprVariant};
use crate::frontend::macros::MacroEntry;
use crate::frontend::{Downstream, Frontend, Ident, SemanticToken, StringRef};

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

pub(crate) trait BuildValue<'a>
where
    Self: Span + StringRef,
    Self: HasValue<<Self as Span>::Span>,
{
    type Builder: backend::ValueBuilder<Ident<Self::StringRef>, Self::Span>
        + DelegateDiagnostics<Self::Span>;

    fn build_value(&'a mut self) -> Self::Builder;
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
pub(crate) struct ValueContext<'a, D: 'a> {
    diagnostics: &'a mut D,
}

#[cfg(test)]
impl<'a, D: 'a> ValueContext<'a, D> {
    pub fn new(diagnostics: &'a mut D) -> Self {
        ValueContext { diagnostics }
    }
}

#[cfg(test)]
impl<'a, D, S> DelegateDiagnostics<S> for ValueContext<'a, D>
where
    D: DownstreamDiagnostics<S> + 'a,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.diagnostics
    }
}

#[cfg(test)]
impl<'a, D, S> HasValue<S> for ValueContext<'a, D>
where
    D: 'a,
    S: Clone,
{
    type Value = RelocExpr<Ident<String>, S>;
}

#[cfg(test)]
impl<'a, D, S> ValueFromSimple<S> for ValueContext<'a, D>
where
    D: 'a,
    S: Clone,
{
    fn from_location_counter(&mut self, span: S) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::LocationCounter, span)
    }

    fn from_number(&mut self, n: i32, span: S) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::Literal(n), span)
    }
}

#[cfg(test)]
impl<'a, D, S> ApplyBinaryOperator<S> for ValueContext<'a, D>
where
    D: 'a,
    S: Clone,
{
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

#[cfg(test)]
impl<'a, D, S> ValueBuilder<Ident<String>, S> for ValueContext<'a, D>
where
    D: 'a,
    S: Clone,
{
    fn from_ident(&mut self, ident: Ident<String>, span: S) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::Symbol(ident), span)
    }
}
