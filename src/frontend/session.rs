use crate::backend::{ApplyBinaryOperator, Backend, BuildValue, HasValue, NameTable, ToValue};
use crate::codebase::CodebaseError;
use crate::diag::span::Span;
use crate::diag::{DelegateDiagnostics, Diagnostics, DownstreamDiagnostics};
use crate::expr::BinaryOperator;
use crate::frontend::macros::MacroEntry;
use crate::frontend::{Downstream, Frontend, Ident, SemanticToken};

pub(super) type MacroArgs<I, S> = Vec<Vec<(SemanticToken<I>, S)>>;

pub(crate) struct CompositeSession<'a, F, B: ?Sized, N, D> {
    pub frontend: &'a mut F,
    pub backend: &'a mut B,
    pub names: &'a mut N,
    pub diagnostics: &'a mut D,
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

impl<'a, F, B, N, D> CompositeSession<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    N: NameTable<Ident<F::StringRef>, MacroEntry = MacroEntry<F, D>>,
    D: Diagnostics,
{
    pub fn define_macro(
        &mut self,
        name: (Ident<F::StringRef>, D::Span),
        params: (Vec<Ident<F::StringRef>>, Vec<D::Span>),
        body: (Vec<SemanticToken<F::StringRef>>, Vec<D::Span>),
    ) {
        self.frontend
            .define_macro(name, params, body, downstream!(self))
    }
}

impl<'a, F, B, N, D: DownstreamDiagnostics<S>, S> DelegateDiagnostics<S>
    for CompositeSession<'a, F, B, N, D>
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.diagnostics
    }
}

impl<'a, F, B, N, D> CompositeSession<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    N: NameTable<Ident<F::StringRef>, MacroEntry = MacroEntry<F, D>>,
    D: Diagnostics,
{
    pub fn invoke_macro(
        &mut self,
        name: (Ident<F::StringRef>, D::Span),
        args: MacroArgs<F::StringRef, D::Span>,
    ) {
        self.frontend.invoke_macro(name, args, downstream!(self))
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
