use crate::backend::{
    ApplyBinaryOperator, Backend, BuildValue, HasValue, Item, NameTable, PartialBackend, ToValue,
};
use crate::codebase::CodebaseError;
use crate::diag::span::Span;
use crate::diag::{DelegateDiagnostics, Diagnostics, DownstreamDiagnostics};
use crate::expr::BinaryOperator;
use crate::frontend::macros::MacroEntry;
use crate::frontend::{Downstream, Frontend, Ident, SemanticToken, StringRef};

pub(crate) trait Session
where
    Self: Span + StringRef,
    Self: DelegateDiagnostics<<Self as Span>::Span>,
    Self: PartialBackend<<Self as Span>::Span>,
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

pub(crate) struct ValueContext<'a, B, D: 'a> {
    builder: B,
    diagnostics: &'a mut D,
}

impl<'a, B, D: 'a> ValueContext<'a, B, D> {
    pub fn new(builder: B, diagnostics: &'a mut D) -> Self {
        ValueContext {
            builder,
            diagnostics,
        }
    }
}

impl<'a, B, D, S> DelegateDiagnostics<S> for ValueContext<'a, B, D>
where
    D: DownstreamDiagnostics<S> + 'a,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.diagnostics
    }
}

impl<'a, B, D, S> HasValue<S> for ValueContext<'a, B, D>
where
    B: HasValue<S>,
    D: 'a,
    S: Clone,
{
    type Value = B::Value;
}

impl<'a, B, D, T, S> ToValue<T, S> for ValueContext<'a, B, D>
where
    B: ToValue<T, S> + 'a,
    D: 'a,
    S: Clone,
{
    fn to_value(&mut self, atom: (T, S)) -> Self::Value {
        self.builder.to_value(atom)
    }
}

impl<'a, B, D, S> ApplyBinaryOperator<S> for ValueContext<'a, B, D>
where
    B: ApplyBinaryOperator<S> + 'a,
    D: 'a,
    S: Clone,
{
    fn apply_binary_operator(
        &mut self,
        operator: (BinaryOperator, S),
        left: Self::Value,
        right: Self::Value,
    ) -> Self::Value {
        self.builder.apply_binary_operator(operator, left, right)
    }
}

type SessionValueContext<'a, F, B, N, D> = ValueContext<
    'a,
    <B as BuildValue<'a, Ident<<F as Frontend<D>>::StringRef>, N, <D as Span>::Span>>::Builder,
    D,
>;

impl<'a, F, B, N, D> CompositeSession<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    N: NameTable<Ident<F::StringRef>, MacroEntry = MacroEntry<F, D>>,
    D: Diagnostics,
{
    pub fn value_context(&mut self) -> SessionValueContext<F, B, N, D> {
        ValueContext::new(
            self.backend.build_value(&mut self.names),
            &mut self.diagnostics,
        )
    }
}
