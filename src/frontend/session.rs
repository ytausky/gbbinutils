use crate::backend;
use crate::backend::{
    ApplyBinaryOperator, Backend, HasValue, Item, Name, NameTable, PartialBackend, ValueFromSimple,
};
use crate::codebase::CodebaseError;
use crate::diag::span::Span;
use crate::diag::{
    CompactDiagnostic, DelegateDiagnostics, Diagnostics, DownstreamDiagnostics, Message,
};
use crate::expr::BinaryOperator;
use crate::frontend::macros::{DefineMacro, Expand, MacroEntry};
use crate::frontend::{Downstream, Frontend, Ident, SemanticToken, StringRef};

#[cfg(test)]
pub(crate) use self::mock::*;

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
        self.names
            .define_macro(name, params, body, self.diagnostics)
    }

    fn invoke_macro(
        &mut self,
        name: (Ident<Self::StringRef>, Self::Span),
        args: MacroArgs<Self::StringRef, Self::Span>,
    ) {
        let expansion = match self.names.get(&name.0) {
            Some(Name::Macro(entry)) => Some(entry.expand(name.1, args, self.diagnostics)),
            Some(_) => unimplemented!(),
            None => {
                let stripped = self.diagnostics.strip_span(&name.1);
                self.diagnostics.emit_diagnostic(CompactDiagnostic::new(
                    Message::UndefinedMacro { name: stripped },
                    name.1,
                ));
                None
            }
        };
        if let Some(expansion) = expansion {
            self.frontend
                .analyze_token_seq(expansion.map(|(t, s)| (Ok(t), s)), &mut downstream!(self))
        }
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

    #[derive(Debug, PartialEq)]
    pub(crate) enum Event {
        AnalyzeFile(String),
        DefineMacro(
            Ident<String>,
            Vec<Ident<String>>,
            Vec<SemanticToken<String>>,
        ),
        InvokeMacro(Ident<String>, Vec<Vec<SemanticToken<String>>>),
    }

    pub(crate) struct MockSession<'a, T, S> {
        log: &'a RefCell<Vec<T>>,
        error: Option<CodebaseError>,
        diagnostics: MockDiagnostics<'a, T, S>,
    }

    impl<'a, T, S> MockSession<'a, T, S> {
        pub fn new(log: &'a RefCell<Vec<T>>) -> Self {
            Self {
                log,
                error: None,
                diagnostics: MockDiagnostics::new(log),
            }
        }

        pub fn fail(&mut self, error: CodebaseError) {
            self.error = Some(error)
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

    impl<'a, T, S: Clone + MockSpan> Span for MockSession<'a, T, S> {
        type Span = S;
    }

    impl<'a, T, S> StringRef for MockSession<'a, T, S> {
        type StringRef = String;
    }

    impl<'a, T, S> Session for MockSession<'a, T, S>
    where
        T: From<Event>,
        T: From<backend::Event<RelocExpr<Ident<String>, S>>>,
        T: From<diag::Event<S>>,
        S: Clone + MockSpan,
    {
        fn analyze_file(&mut self, path: String) -> Result<(), CodebaseError> {
            self.log.borrow_mut().push(Event::AnalyzeFile(path).into());
            self.error.take().map_or(Ok(()), Err)
        }

        fn define_macro(
            &mut self,
            name: (Ident<Self::StringRef>, Self::Span),
            params: (Vec<Ident<Self::StringRef>>, Vec<Self::Span>),
            body: (Vec<SemanticToken<Self::StringRef>>, Vec<Self::Span>),
        ) {
            self.log
                .borrow_mut()
                .push(Event::DefineMacro(name.0, params.0, body.0).into())
        }

        fn invoke_macro(
            &mut self,
            name: (Ident<Self::StringRef>, Self::Span),
            args: MacroArgs<Self::StringRef, Self::Span>,
        ) {
            self.log.borrow_mut().push(
                Event::InvokeMacro(
                    name.0,
                    args.into_iter()
                        .map(|arg| arg.into_iter().map(|(token, _)| token).collect())
                        .collect(),
                )
                .into(),
            )
        }

        fn define_symbol(
            &mut self,
            symbol: (Ident<Self::StringRef>, Self::Span),
            value: Self::Value,
        ) {
            self.log
                .borrow_mut()
                .push(backend::Event::DefineSymbol(symbol, value).into())
        }
    }

    impl<'a, T, S> PartialBackend<S> for MockSession<'a, T, S>
    where
        T: From<backend::Event<RelocExpr<Ident<String>, S>>>,
        S: Clone + MockSpan,
    {
        fn emit_item(&mut self, item: Item<Self::Value>) {
            self.log
                .borrow_mut()
                .push(backend::Event::EmitItem(item).into())
        }

        fn set_origin(&mut self, origin: Self::Value) {
            self.log
                .borrow_mut()
                .push(backend::Event::SetOrigin(origin).into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend;
    use crate::backend::{HashMapNameTable, RelocExpr};
    use crate::diag;
    use crate::diag::MockSpan;
    use crate::frontend;
    use crate::frontend::{FrontendEvent, Literal};
    use crate::syntax::{Command, Directive, Mnemonic, Token};

    use std::cell::RefCell;
    use std::iter;

    #[test]
    fn define_and_invoke_macro() {
        let name = "my_macro";
        let tokens = vec![Token::Command(Command::Mnemonic(Mnemonic::Nop))];
        let spans: Vec<_> = iter::repeat(()).take(tokens.len()).collect();
        let log = RefCell::new(Vec::new());
        let mut fixture = Fixture::new(&log);
        let mut session = fixture.session();
        session.define_macro(
            (name.into(), ()),
            (vec![], vec![]),
            (tokens.clone(), spans.clone()),
        );
        session.invoke_macro((name.into(), ()), vec![]);
        assert_eq!(
            *log.borrow(),
            [FrontendEvent::AnalyzeTokenSeq(
                tokens.into_iter().map(|token| (Ok(token), ())).collect()
            )
            .into()]
        );
    }

    #[test]
    fn define_and_invoke_macro_with_param() {
        let db = Token::Command(Command::Directive(Directive::Db));
        let arg = Token::Literal(Literal::Number(0x42));
        let literal0 = Token::Literal(Literal::Number(0));
        let log = RefCell::new(Vec::new());
        let mut fixture = Fixture::new(&log);
        let mut session = fixture.session();
        let name = "my_db";
        let param = "x";
        session.define_macro(
            (name.into(), ()),
            (vec![param.into()], vec![()]),
            (
                vec![db.clone(), Token::Ident(param.into()), literal0.clone()],
                vec![(), (), ()],
            ),
        );
        session.invoke_macro((name.into(), ()), vec![vec![(arg.clone(), ())]]);
        assert_eq!(
            *log.borrow(),
            [FrontendEvent::AnalyzeTokenSeq(
                vec![db, arg, literal0]
                    .into_iter()
                    .map(|token| (Ok(token), ()))
                    .collect()
            )
            .into()]
        );
    }

    #[test]
    fn define_and_invoke_macro_with_label() {
        let nop = Token::Command(Command::Mnemonic(Mnemonic::Nop));
        let label = "label";
        let log = RefCell::new(Vec::new());
        let mut fixture = Fixture::new(&log);
        let mut session = fixture.session();
        let name = "my_macro";
        let param = "x";
        session.define_macro(
            (name.into(), ()),
            (vec![param.into()], vec![()]),
            (vec![Token::Label(param.into()), nop.clone()], vec![(), ()]),
        );
        session.invoke_macro(
            (name.into(), ()),
            vec![vec![(Token::Ident(label.into()), ())]],
        );
        assert_eq!(
            *log.borrow(),
            [FrontendEvent::AnalyzeTokenSeq(
                vec![Token::Label(label.into()), nop]
                    .into_iter()
                    .map(|token| (Ok(token), ()))
                    .collect()
            )
            .into()]
        );
    }

    #[test]
    fn diagnose_undefined_macro() {
        let name = "my_macro";
        let log = RefCell::new(Vec::new());
        let mut fixture = Fixture::new(&log);
        let mut session = fixture.session();
        session.invoke_macro((name.into(), name), vec![]);
        assert_eq!(
            *log.borrow(),
            [diag::Event::EmitDiagnostic(CompactDiagnostic::new(
                Message::UndefinedMacro { name },
                name
            ))
            .into()]
        );
    }

    type MockFrontend<'a, S> = frontend::MockFrontend<'a, Event<S>>;
    type MockBackend<'a, S> = backend::MockBackend<'a, Event<S>>;
    type MockDiagnostics<'a, S> = diag::MockDiagnostics<'a, Event<S>, S>;
    type TestNameTable<'a, S> =
        HashMapNameTable<MacroEntry<MockFrontend<'a, S>, MockDiagnostics<'a, S>>>;
    type TestSession<'a, 'b, S> = CompositeSession<
        'b,
        MockFrontend<'a, S>,
        MockBackend<'a, S>,
        TestNameTable<'a, S>,
        MockDiagnostics<'a, S>,
    >;

    #[derive(Debug, PartialEq)]
    enum Event<S: Clone> {
        Frontend(FrontendEvent<S>),
        Backend(backend::Event<RelocExpr<Ident<String>, S>>),
        Diagnostics(diag::Event<S>),
    }

    impl<S: Clone> From<FrontendEvent<S>> for Event<S> {
        fn from(event: FrontendEvent<S>) -> Self {
            Event::Frontend(event)
        }
    }

    impl<S: Clone> From<backend::Event<RelocExpr<Ident<String>, S>>> for Event<S> {
        fn from(event: backend::Event<RelocExpr<Ident<String>, S>>) -> Self {
            Event::Backend(event)
        }
    }

    impl<S: Clone> From<diag::Event<S>> for Event<S> {
        fn from(event: diag::Event<S>) -> Self {
            Event::Diagnostics(event)
        }
    }

    struct Fixture<'a, S: Clone + MockSpan> {
        frontend: MockFrontend<'a, S>,
        backend: MockBackend<'a, S>,
        names: TestNameTable<'a, S>,
        diagnostics: MockDiagnostics<'a, S>,
    }

    impl<'a, S: Clone + MockSpan> Fixture<'a, S> {
        fn new(log: &'a RefCell<Vec<Event<S>>>) -> Self {
            Self {
                frontend: MockFrontend::new(log),
                backend: MockBackend::new(log),
                names: HashMapNameTable::new(),
                diagnostics: MockDiagnostics::new(log),
            }
        }

        fn session<'b>(&'b mut self) -> TestSession<'a, 'b, S> {
            CompositeSession::new(
                &mut self.frontend,
                &mut self.backend,
                &mut self.names,
                &mut self.diagnostics,
            )
        }
    }
}
