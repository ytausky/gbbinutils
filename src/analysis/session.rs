use super::backend::*;
use super::macros::{DefineMacro, Expand, MacroEntry};
use super::semantics::Analyze;
use super::{Ident, Lex, SemanticToken, StringRef};

use crate::codebase::CodebaseError;
use crate::diag::span::Span;
use crate::diag::*;
use crate::expr::BinaryOperator;
use crate::model::Item;

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
    Self: ValueFromSimple<S>,
    Self: ApplyBinaryOperator<S>,
{
    fn from_ident(&mut self, ident: I, span: S) -> Self::Value;
}

pub(super) type MacroArgs<I, S> = Vec<Vec<(SemanticToken<I>, S)>>;

pub(crate) struct CompositeSession<'a, C, A, B: ?Sized, N, D> {
    codebase: &'a mut C,
    analyzer: &'a mut A,
    backend: &'a mut B,
    names: &'a mut N,
    diagnostics: &'a mut D,
}

impl<'a, C, A, B: ?Sized, N, D> CompositeSession<'a, C, A, B, N, D> {
    pub fn new(
        codebase: &'a mut C,
        analyzer: &'a mut A,
        backend: &'a mut B,
        names: &'a mut N,
        diagnostics: &'a mut D,
    ) -> CompositeSession<'a, C, A, B, N, D> {
        CompositeSession {
            codebase,
            analyzer,
            backend,
            names,
            diagnostics,
        }
    }
}

pub struct PartialSession<'a, C: 'a, B: ?Sized + 'a, N: 'a, D: 'a> {
    pub codebase: &'a mut C,
    pub backend: &'a mut B,
    pub names: &'a mut N,
    pub diagnostics: &'a mut D,
}

macro_rules! partial {
    ($session:expr) => {
        PartialSession {
            codebase: $session.codebase,
            backend: $session.backend,
            names: $session.names,
            diagnostics: $session.diagnostics,
        }
    };
}

impl<'a, F, A, B, N, D> Span for CompositeSession<'a, F, A, B, N, D>
where
    B: ?Sized,
    D: Span,
{
    type Span = D::Span;
}

impl<'a, C, A, B, N, D> StringRef for CompositeSession<'a, C, A, B, N, D>
where
    C: Lex<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type StringRef = C::StringRef;
}

impl<'a, F, A, B, N, D> HasValue<D::Span> for CompositeSession<'a, F, A, B, N, D>
where
    B: HasValue<D::Span> + ?Sized,
    D: Span,
{
    type Value = B::Value;
}

impl<'a, C, A, B, N, D> PartialBackend<D::Span> for CompositeSession<'a, C, A, B, N, D>
where
    C: Lex<D>,
    B: Backend<Ident<C::StringRef>, D::Span, N> + ?Sized,
    N: NameTable<Ident<C::StringRef>, MacroEntry = MacroEntry<C::StringRef, D>>,
    D: Diagnostics,
{
    fn emit_item(&mut self, item: Item<Self::Value>) {
        self.backend.emit_item(item)
    }

    fn set_origin(&mut self, origin: Self::Value) {
        self.backend.set_origin(origin)
    }
}

impl<'a, C, A, B, N, D> ValueFromSimple<D::Span> for CompositeSession<'a, C, A, B, N, D>
where
    C: Lex<D>,
    B: Backend<Ident<C::StringRef>, D::Span, N> + ?Sized,
    N: NameTable<Ident<C::StringRef>, MacroEntry = MacroEntry<C::StringRef, D>>,
    D: Diagnostics,
{
    fn from_location_counter(&mut self, span: D::Span) -> Self::Value {
        self.backend.from_location_counter(span)
    }

    fn from_number(&mut self, n: i32, span: D::Span) -> Self::Value {
        self.backend.from_number(n, span)
    }
}

impl<'a, C, A, B, N, D> ApplyBinaryOperator<D::Span> for CompositeSession<'a, C, A, B, N, D>
where
    C: Lex<D>,
    B: Backend<Ident<C::StringRef>, D::Span, N> + ?Sized,
    N: NameTable<Ident<C::StringRef>, MacroEntry = MacroEntry<C::StringRef, D>>,
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

impl<'a, C, A, B, N, D> ValueBuilder<Ident<C::StringRef>, D::Span>
    for CompositeSession<'a, C, A, B, N, D>
where
    C: Lex<D>,
    B: Backend<Ident<C::StringRef>, D::Span, N> + ?Sized,
    N: NameTable<Ident<C::StringRef>, MacroEntry = MacroEntry<C::StringRef, D>>,
    D: Diagnostics,
{
    fn from_ident(&mut self, ident: Ident<C::StringRef>, span: D::Span) -> Self::Value {
        self.backend.from_ident(ident, span, self.names)
    }
}

impl<'a, C, A, B, N, D> Session for CompositeSession<'a, C, A, B, N, D>
where
    C: Lex<D>,
    A: Analyze<C::StringRef, D>,
    B: Backend<Ident<C::StringRef>, D::Span, N> + ?Sized,
    N: NameTable<Ident<C::StringRef>, MacroEntry = MacroEntry<C::StringRef, D>>,
    D: Diagnostics,
{
    fn analyze_file(&mut self, path: Self::StringRef) -> Result<(), CodebaseError> {
        let tokens = self.codebase.lex_file(path, self.diagnostics)?;
        self.analyzer.analyze_token_seq(tokens, &mut partial!(self));
        Ok(())
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
            self.analyzer
                .analyze_token_seq(expansion.map(|(t, s)| (Ok(t), s)), &mut partial!(self))
        }
    }

    fn define_symbol(&mut self, symbol: (Ident<Self::StringRef>, Self::Span), value: Self::Value) {
        self.backend.define_symbol(symbol, value, &mut self.names)
    }
}

impl<'a, F, A, B, N, D, S> DelegateDiagnostics<S> for CompositeSession<'a, F, A, B, N, D>
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

    use crate::analysis::backend::BackendEvent;
    use crate::diag::{DiagnosticsEvent, MockDiagnostics, MockSpan};
    use crate::expr::{Expr, ExprVariant};
    use crate::model::{RelocAtom, RelocExpr};

    use std::cell::RefCell;

    #[derive(Debug, PartialEq)]
    pub(crate) enum SessionEvent {
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
        T: From<DiagnosticsEvent<S>>,
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
        T: From<SessionEvent>,
        T: From<BackendEvent<RelocExpr<Ident<String>, S>>>,
        T: From<DiagnosticsEvent<S>>,
        S: Clone + MockSpan,
    {
        fn analyze_file(&mut self, path: String) -> Result<(), CodebaseError> {
            self.log
                .borrow_mut()
                .push(SessionEvent::AnalyzeFile(path).into());
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
                .push(SessionEvent::DefineMacro(name.0, params.0, body.0).into())
        }

        fn invoke_macro(
            &mut self,
            name: (Ident<Self::StringRef>, Self::Span),
            args: MacroArgs<Self::StringRef, Self::Span>,
        ) {
            self.log.borrow_mut().push(
                SessionEvent::InvokeMacro(
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
                .push(BackendEvent::DefineSymbol(symbol, value).into())
        }
    }

    impl<'a, T, S> PartialBackend<S> for MockSession<'a, T, S>
    where
        T: From<BackendEvent<RelocExpr<Ident<String>, S>>>,
        S: Clone + MockSpan,
    {
        fn emit_item(&mut self, item: Item<Self::Value>) {
            self.log
                .borrow_mut()
                .push(BackendEvent::EmitItem(item).into())
        }

        fn set_origin(&mut self, origin: Self::Value) {
            self.log
                .borrow_mut()
                .push(BackendEvent::SetOrigin(origin).into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::backend::{BackendEvent, BasicNameTable};
    use crate::analysis::semantics::AnalyzerEvent;
    use crate::analysis::{Literal, MockCodebase};
    use crate::diag::{DiagnosticsEvent, MockSpan};
    use crate::model::{Instruction, Nullary, RelocAtom, RelocExpr};
    use crate::syntax::{Command, Directive, Mnemonic, Token};

    use std::cell::RefCell;
    use std::iter;

    #[test]
    fn emit_instruction_item() {
        let item = Item::Instruction(Instruction::Nullary(Nullary::Nop));
        let log = RefCell::new(Vec::new());
        let mut fixture = Fixture::<()>::new(&log);
        let mut session = fixture.session();
        session.emit_item(item.clone());
        assert_eq!(log.into_inner(), [BackendEvent::EmitItem(item).into()]);
    }

    #[test]
    fn define_label() {
        let label = "label";
        let log = RefCell::new(Vec::new());
        let mut fixture = Fixture::new(&log);
        let mut session = fixture.session();
        session.define_symbol((label.into(), ()), RelocAtom::LocationCounter.into());
        assert_eq!(
            log.into_inner(),
            [
                BackendEvent::DefineSymbol((label.into(), ()), RelocAtom::LocationCounter.into())
                    .into()
            ]
        );
    }

    #[test]
    fn include_source_file() {
        let path = "my_file.s";
        let tokens = vec![(Ok(Token::Command(Command::Mnemonic(Mnemonic::Nop))), ())];
        let log = RefCell::new(Vec::new());
        let mut fixture = Fixture::new(&log);
        fixture.codebase.set_file(path, tokens.clone());
        let mut session = fixture.session();
        session.analyze_file(path.into()).unwrap();
        assert_eq!(
            log.into_inner(),
            [AnalyzerEvent::AnalyzeTokenSeq(tokens).into()]
        );
    }

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
            log.into_inner(),
            [AnalyzerEvent::AnalyzeTokenSeq(
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
            log.into_inner(),
            [AnalyzerEvent::AnalyzeTokenSeq(
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
            log.into_inner(),
            [AnalyzerEvent::AnalyzeTokenSeq(
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
            log.into_inner(),
            [DiagnosticsEvent::EmitDiagnostic(CompactDiagnostic::new(
                Message::UndefinedMacro { name },
                name
            ))
            .into()]
        );
    }

    type MockAnalyzer<'a, S> = crate::analysis::semantics::MockAnalyzer<'a, Event<S>>;
    type MockBackend<'a, S> = crate::analysis::backend::MockBackend<'a, Event<S>>;
    type MockDiagnostics<'a, S> = crate::diag::MockDiagnostics<'a, Event<S>, S>;
    type TestNameTable<'a, S> = BasicNameTable<MacroEntry<String, MockDiagnostics<'a, S>>, ()>;
    type TestSession<'a, 'b, S> = CompositeSession<
        'b,
        MockCodebase<S>,
        MockAnalyzer<'a, S>,
        MockBackend<'a, S>,
        TestNameTable<'a, S>,
        MockDiagnostics<'a, S>,
    >;

    #[derive(Debug, PartialEq)]
    enum Event<S: Clone> {
        Frontend(AnalyzerEvent<S>),
        Backend(BackendEvent<RelocExpr<Ident<String>, S>>),
        Diagnostics(DiagnosticsEvent<S>),
    }

    impl<S: Clone> From<AnalyzerEvent<S>> for Event<S> {
        fn from(event: AnalyzerEvent<S>) -> Self {
            Event::Frontend(event)
        }
    }

    impl<S: Clone> From<BackendEvent<RelocExpr<Ident<String>, S>>> for Event<S> {
        fn from(event: BackendEvent<RelocExpr<Ident<String>, S>>) -> Self {
            Event::Backend(event)
        }
    }

    impl<S: Clone> From<DiagnosticsEvent<S>> for Event<S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            Event::Diagnostics(event)
        }
    }

    struct Fixture<'a, S: Clone + MockSpan> {
        codebase: MockCodebase<S>,
        analyzer: MockAnalyzer<'a, S>,
        backend: MockBackend<'a, S>,
        names: TestNameTable<'a, S>,
        diagnostics: MockDiagnostics<'a, S>,
    }

    impl<'a, S: Clone + MockSpan> Fixture<'a, S> {
        fn new(log: &'a RefCell<Vec<Event<S>>>) -> Self {
            Self {
                codebase: MockCodebase::new(),
                analyzer: MockAnalyzer::new(log),
                backend: MockBackend::new(log),
                names: BasicNameTable::new(),
                diagnostics: MockDiagnostics::new(log),
            }
        }

        fn session<'b>(&'b mut self) -> TestSession<'a, 'b, S> {
            CompositeSession::new(
                &mut self.codebase,
                &mut self.analyzer,
                &mut self.backend,
                &mut self.names,
                &mut self.diagnostics,
            )
        }
    }

    impl MockSpan for &'static str {
        fn default() -> Self {
            unimplemented!()
        }

        fn merge(&self, _: &Self) -> Self {
            unimplemented!()
        }
    }
}
