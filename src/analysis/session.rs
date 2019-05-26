pub use super::backend::ValueBuilder;

use super::backend::*;
use super::macros::{DefineMacro, Expand, MacroEntry};
use super::resolve::{Ident, Name, NameTable, StartScope};
use super::semantics::Analyze;
use super::{Lex, SemanticToken, StringSource};

use crate::codebase::CodebaseError;
use crate::diag::span::SpanSource;
use crate::diag::*;
use crate::model::{BinOp, FnCall, Item, ParamId};

#[cfg(test)]
pub(crate) use self::mock::*;

pub(super) trait Session
where
    Self: SpanSource + StringSource,
    Self: BasicSession<<Self as StringSource>::StringRef, <Self as SpanSource>::Span>,
{
    fn analyze_file(self, path: Self::StringRef) -> (Result<(), CodebaseError>, Self);

    fn define_macro(
        &mut self,
        name: (Ident<Self::StringRef>, Self::Span),
        params: Params<Self::StringRef, Self::Span>,
        body: (Vec<SemanticToken<Self::StringRef>>, Vec<Self::Span>),
    );
    fn call_macro(
        self,
        name: (Ident<Self::StringRef>, Self::Span),
        args: MacroArgs<Self::StringRef, Self::Span>,
    ) -> Self;
}

pub(super) trait BasicSession<R, S: Clone>
where
    Self: Sized,
    Self: PartialBackend<S>,
    Self: StartSection<Ident<R>, S>,
    Self: Diagnostics<S>,
{
    type FnBuilder: ValueBuilder<Ident<R>, S> + FinishFnDef<Return = Self> + Diagnostics<S>;
    type GeneralBuilder: ValueBuilder<Ident<R>, S>
        + Finish<S, Parent = Self, Value = Self::Value>
        + Diagnostics<S>;

    fn build_value(self) -> Self::GeneralBuilder;
    fn define_symbol(self, name: Ident<R>, span: S) -> Self::FnBuilder;
}

pub(super) type MacroArgs<I, S> = Vec<Vec<(SemanticToken<I>, S)>>;
pub(super) type Params<R, S> = (Vec<Ident<R>>, Vec<S>);

pub(super) struct CompositeSession<'a, 'b, C, A, B, N, D> {
    upstream: Upstream<'a, 'b, C, A>,
    downstream: Downstream<B, &'a mut N, Wrapper<'a, D>>,
}

pub(super) struct Upstream<'a, 'b, C, A> {
    codebase: &'a mut C,
    analyzer: &'b mut A,
}

pub(super) struct Downstream<B, N, D> {
    backend: B,
    names: N,
    diagnostics: D,
}

pub(super) struct Wrapper<'a, D>(&'a mut D);

impl<'a, 'b, C, A, B, N, D> CompositeSession<'a, 'b, C, A, B, N, D> {
    pub fn new(
        codebase: &'a mut C,
        analyzer: &'b mut A,
        backend: B,
        names: &'a mut N,
        diagnostics: &'a mut D,
    ) -> Self {
        CompositeSession {
            upstream: Upstream { codebase, analyzer },
            downstream: Downstream {
                backend,
                names,
                diagnostics: Wrapper(diagnostics),
            },
        }
    }
}

impl<'a, B, N, D> Downstream<B, &'a mut N, Wrapper<'a, D>> {
    fn look_up_symbol<R, S>(&mut self, ident: Ident<R>, span: &S) -> B::Name
    where
        B: AllocName<S>,
        N: NameTable<Ident<R>, BackendEntry = B::Name>,
        D: Diagnostics<S>,
        S: Clone,
    {
        match self.names.get(&ident) {
            Some(Name::Backend(id)) => id.clone(),
            Some(Name::Macro(_)) => {
                self.diagnostics
                    .0
                    .emit_diag(Message::MacroNameInExpr.at(span.clone()));
                self.backend.alloc_name(span.clone())
            }
            None => {
                let id = self.backend.alloc_name(span.clone());
                self.names.insert(ident, Name::Backend(id.clone()));
                id
            }
        }
    }

    fn replace_backend<T>(
        self,
        f: impl FnOnce(B) -> T,
    ) -> Downstream<T, &'a mut N, Wrapper<'a, D>> {
        Downstream {
            backend: f(self.backend),
            names: self.names,
            diagnostics: self.diagnostics,
        }
    }
}

pub(super) struct PartialSession<'a, C, B, N, D> {
    codebase: &'a mut C,
    downstream: Downstream<B, &'a mut N, Wrapper<'a, D>>,
}

macro_rules! partial {
    ($session:expr) => {
        PartialSession {
            codebase: $session.upstream.codebase,
            downstream: Downstream {
                backend: $session.downstream.backend,
                names: $session.downstream.names,
                diagnostics: Wrapper(&mut *$session.downstream.diagnostics.0),
            },
        }
    };
}

impl<'a, C, B, N, D> PartialSession<'a, C, B, N, D> {
    pub fn into_session<A>(self, analyzer: &mut A) -> CompositeSession<'a, '_, C, A, B, N, D> {
        CompositeSession {
            upstream: Upstream {
                codebase: self.codebase,
                analyzer,
            },
            downstream: self.downstream,
        }
    }
}

impl<'a, 'b, C, A, B, N, D> From<CompositeSession<'a, 'b, C, A, B, N, D>>
    for PartialSession<'a, C, B, N, D>
{
    fn from(session: CompositeSession<'a, 'b, C, A, B, N, D>) -> Self {
        partial!(session)
    }
}

impl<'a, 'b, F, A, B, N, D> SpanSource for CompositeSession<'a, 'b, F, A, B, N, D>
where
    D: SpanSource,
{
    type Span = D::Span;
}

impl<'a, 'b, C: StringSource, A, B, N, D> StringSource for CompositeSession<'a, 'b, C, A, B, N, D> {
    type StringRef = C::StringRef;
}

impl<'a, 'b, C, A, B, N, D, S> PartialBackend<S> for CompositeSession<'a, 'b, C, A, B, N, D>
where
    B: Backend<S>,
    S: Clone,
{
    type Value = B::Value;

    fn emit_item(&mut self, item: Item<Self::Value>) {
        self.downstream.backend.emit_item(item)
    }

    fn reserve(&mut self, bytes: Self::Value) {
        self.downstream.backend.reserve(bytes)
    }

    fn set_origin(&mut self, origin: Self::Value) {
        self.downstream.backend.set_origin(origin)
    }
}

macro_rules! impl_push_op_for_downstream {
    ($t:ty) => {
        impl<B, N, D, S> PushOp<$t, S> for Downstream<B, N, D>
        where
            B: PushOp<$t, S>,
            S: Clone,
        {
            fn push_op(&mut self, op: $t, span: S) {
                self.backend.push_op(op, span)
            }
        }
    };
}

impl_push_op_for_downstream! {LocationCounter}
impl_push_op_for_downstream! {i32}
impl_push_op_for_downstream! {BinOp}
impl_push_op_for_downstream! {ParamId}
impl_push_op_for_downstream! {FnCall}

impl<'a, 'b, C, A, B, N, D, R, S> PushOp<Ident<R>, S>
    for RelocContext<Upstream<'a, 'b, C, A>, Downstream<B, &'a mut N, Wrapper<'a, D>>>
where
    B: AllocName<S> + PushOp<<B as AllocName<S>>::Name, S>,
    N: NameTable<Ident<R>, BackendEntry = B::Name>,
    D: Diagnostics<S>,
    S: Clone,
{
    fn push_op(&mut self, ident: Ident<R>, span: S) {
        let id = self.builder.look_up_symbol(ident, &span);
        self.builder.backend.push_op(id, span)
    }
}

impl<'a, 'b, C, A, B, N, D, S> Finish<S>
    for RelocContext<Upstream<'a, 'b, C, A>, Downstream<B, &'a mut N, Wrapper<'a, D>>>
where
    B: Finish<S>,
    S: Clone,
{
    type Parent = CompositeSession<'a, 'b, C, A, B::Parent, N, D>;
    type Value = B::Value;

    fn finish(self) -> (Self::Parent, Self::Value) {
        let (backend, value) = self.builder.backend.finish();
        let parent = CompositeSession {
            upstream: self.parent,
            downstream: Downstream {
                backend,
                names: self.builder.names,
                diagnostics: self.builder.diagnostics,
            },
        };
        (parent, value)
    }
}

impl<'a, 'b, C, A, B, N, D> FinishFnDef
    for RelocContext<Upstream<'a, 'b, C, A>, Downstream<B, &'a mut N, Wrapper<'a, D>>>
where
    B: FinishFnDef,
{
    type Return = CompositeSession<'a, 'b, C, A, B::Return, N, D>;

    fn finish_fn_def(self) -> Self::Return {
        CompositeSession {
            upstream: self.parent,
            downstream: self.builder.replace_backend(FinishFnDef::finish_fn_def),
        }
    }
}

delegate_diagnostics! {
    {P, B, N, D: Diagnostics<S>, S},
    RelocContext<P, Downstream<B, N, D>>,
    {builder.diagnostics},
    D,
    S
}

delegate_diagnostics! {
    {'a, D: Diagnostics<S>, S},
    Wrapper<'a, D>,
    {0},
    D,
    S
}

impl<'a, 'b, C, A, B, N, D> Session for CompositeSession<'a, 'b, C, A, B, N, D>
where
    C: Lex<D>,
    A: Analyze<C::StringRef, D>,
    B: Backend<D::Span>,
    N: NameTable<
            Ident<C::StringRef>,
            BackendEntry = B::Name,
            MacroEntry = MacroEntry<C::StringRef, D>,
        > + StartScope<Ident<C::StringRef>>,
    D: DiagnosticsSystem,
{
    fn analyze_file(mut self, path: Self::StringRef) -> (Result<(), CodebaseError>, Self) {
        let tokens = match self
            .upstream
            .codebase
            .lex_file(path, &mut *self.downstream.diagnostics.0)
        {
            Ok(tokens) => tokens,
            Err(error) => return (Err(error), self),
        };
        let PartialSession {
            downstream: Downstream { backend, .. },
            ..
        } = self
            .upstream
            .analyzer
            .analyze_token_seq(tokens, partial!(self));
        self.downstream.backend = backend;
        (Ok(()), self)
    }

    fn define_macro(
        &mut self,
        name: (Ident<Self::StringRef>, Self::Span),
        params: (Vec<Ident<Self::StringRef>>, Vec<Self::Span>),
        body: (Vec<SemanticToken<Self::StringRef>>, Vec<Self::Span>),
    ) {
        self.downstream
            .names
            .define_macro(name, params, body, self.downstream.diagnostics.0)
    }

    fn call_macro(
        mut self,
        name: (Ident<Self::StringRef>, Self::Span),
        args: MacroArgs<Self::StringRef, Self::Span>,
    ) -> Self {
        let expansion = match self.downstream.names.get(&name.0) {
            Some(Name::Macro(entry)) => {
                Some(entry.expand(name.1, args, self.downstream.diagnostics.0))
            }
            Some(_) => unimplemented!(),
            None => {
                let stripped = self.downstream.diagnostics.0.strip_span(&name.1);
                self.downstream
                    .diagnostics
                    .0
                    .emit_diag(Message::UndefinedMacro { name: stripped }.at(name.1));
                None
            }
        };
        if let Some(expansion) = expansion {
            let PartialSession {
                downstream: Downstream { backend, .. },
                ..
            } = self
                .upstream
                .analyzer
                .analyze_token_seq(expansion.map(|(t, s)| (Ok(t), s)), partial!(self));
            self.downstream.backend = backend
        }
        self
    }
}

impl<'a, 'b, C, A, B, N, D, R, S> BasicSession<R, S> for CompositeSession<'a, 'b, C, A, B, N, D>
where
    B: Backend<S>,
    N: NameTable<Ident<R>, BackendEntry = B::Name> + StartScope<Ident<R>>,
    D: Diagnostics<S>,
    S: Clone,
{
    type FnBuilder = RelocContext<
        Upstream<'a, 'b, C, A>,
        Downstream<B::SymbolBuilder, &'a mut N, Wrapper<'a, D>>,
    >;
    type GeneralBuilder = RelocContext<
        Upstream<'a, 'b, C, A>,
        Downstream<B::ImmediateBuilder, &'a mut N, Wrapper<'a, D>>,
    >;

    fn build_value(self) -> Self::GeneralBuilder {
        RelocContext {
            parent: self.upstream,
            builder: self.downstream.replace_backend(Backend::build_immediate),
        }
    }

    fn define_symbol(mut self, name: Ident<R>, span: S) -> Self::FnBuilder {
        self.downstream.names.start_scope(&name);
        let id = self.downstream.look_up_symbol(name, &span);
        RelocContext {
            parent: self.upstream,
            builder: self
                .downstream
                .replace_backend(|backend| backend.define_fn(id, span)),
        }
    }
}

delegate_diagnostics! {
    {'a, 'b, F, A, B, N, D: Diagnostics<S>, S},
    CompositeSession<'a, 'b, F, A, B, N, D>,
    {downstream.diagnostics},
    D,
    S
}

impl<'a, 'b, C, A, B, N, D, R, S> StartSection<Ident<R>, S>
    for CompositeSession<'a, 'b, C, A, B, N, D>
where
    B: Backend<S>,
    N: NameTable<Ident<R>, BackendEntry = B::Name>,
    D: Diagnostics<S>,
    S: Clone,
{
    fn start_section(&mut self, (ident, span): (Ident<R>, S)) {
        let name = self.downstream.look_up_symbol(ident, &span);
        self.downstream.backend.start_section((name, span))
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::analysis::backend::{BackendEvent, MockSymbolBuilder};
    use crate::diag::{DiagnosticsEvent, MockDiagnostics};
    use crate::log::Log;

    use std::marker::PhantomData;

    type Expr<S> = crate::model::Expr<LocationCounter, Ident<String>, S>;

    #[derive(Debug, PartialEq)]
    pub(crate) enum SessionEvent<S> {
        AnalyzeFile(String),
        DefineMacro(
            Ident<String>,
            Vec<Ident<String>>,
            Vec<SemanticToken<String>>,
        ),
        InvokeMacro(Ident<String>, Vec<Vec<SemanticToken<String>>>),
        DefineSymbol((Ident<String>, S), Expr<S>),
    }

    pub(in crate::analysis) struct MockSession<T, S> {
        log: Log<T>,
        error: Option<CodebaseError>,
        diagnostics: MockDiagnostics<T>,
        _span: PhantomData<S>,
    }

    impl<T, S> MockSession<T, S> {
        pub fn new(log: Log<T>) -> Self {
            Self {
                log: log.clone(),
                error: None,
                diagnostics: MockDiagnostics::new(log),
                _span: PhantomData,
            }
        }

        pub fn fail(&mut self, error: CodebaseError) {
            self.error = Some(error)
        }
    }

    delegate_diagnostics! {
        {T: From<DiagnosticsEvent<S>>, S: Merge},
        MockSession<T, S>,
        {diagnostics},
        MockDiagnostics<T>,
        S
    }

    impl<T, S: Clone + Merge> SpanSource for MockSession<T, S> {
        type Span = S;
    }

    impl<T, S> StringSource for MockSession<T, S> {
        type StringRef = String;
    }

    impl<T, S> Session for MockSession<T, S>
    where
        T: From<SessionEvent<S>>,
        T: From<BackendEvent<Expr<S>>>,
        T: From<DiagnosticsEvent<S>>,
        S: Clone + Merge,
    {
        fn analyze_file(mut self, path: String) -> (Result<(), CodebaseError>, Self) {
            self.log.push(SessionEvent::AnalyzeFile(path));
            (self.error.take().map_or(Ok(()), Err), self)
        }

        fn define_macro(
            &mut self,
            name: (Ident<Self::StringRef>, Self::Span),
            params: (Vec<Ident<Self::StringRef>>, Vec<Self::Span>),
            body: (Vec<SemanticToken<Self::StringRef>>, Vec<Self::Span>),
        ) {
            self.log
                .push(SessionEvent::DefineMacro(name.0, params.0, body.0))
        }

        fn call_macro(
            self,
            name: (Ident<Self::StringRef>, Self::Span),
            args: MacroArgs<Self::StringRef, Self::Span>,
        ) -> Self {
            self.log.push(SessionEvent::InvokeMacro(
                name.0,
                args.into_iter()
                    .map(|arg| arg.into_iter().map(|(token, _)| token).collect())
                    .collect(),
            ));
            self
        }
    }

    impl<T, S> BasicSession<String, S> for MockSession<T, S>
    where
        T: From<SessionEvent<S>>,
        T: From<BackendEvent<Expr<S>>>,
        T: From<DiagnosticsEvent<S>>,
        S: Clone + Merge,
    {
        type FnBuilder = MockSymbolBuilder<Self, Ident<String>, S>;
        type GeneralBuilder = RelocContext<(), Downstream<Expr<S>, (), Self>>;

        fn build_value(self) -> Self::GeneralBuilder {
            RelocContext {
                parent: (),
                builder: Downstream {
                    backend: Default::default(),
                    names: (),
                    diagnostics: self,
                },
            }
        }

        fn define_symbol(self, name: Ident<String>, span: S) -> Self::FnBuilder {
            MockSymbolBuilder {
                parent: self,
                name: (name, span),
                expr: Default::default(),
            }
        }
    }

    impl<T, S: Clone> Finish<S> for RelocContext<(), Downstream<Expr<S>, (), MockSession<T, S>>> {
        type Parent = MockSession<T, S>;
        type Value = Expr<S>;

        fn finish(self) -> (Self::Parent, Self::Value) {
            (self.builder.diagnostics, self.builder.backend)
        }
    }

    impl<T, S> FinishFnDef for MockSymbolBuilder<MockSession<T, S>, Ident<String>, S>
    where
        T: From<SessionEvent<S>>,
    {
        type Return = MockSession<T, S>;

        fn finish_fn_def(self) -> Self::Return {
            let parent = self.parent;
            parent
                .log
                .push(SessionEvent::DefineSymbol(self.name, self.expr));
            parent
        }
    }

    delegate_diagnostics! {
        {T: From<DiagnosticsEvent<S>>, S: Merge},
        MockSymbolBuilder<MockSession<T, S>, Ident<String>, S>,
        {parent.diagnostics},
        MockDiagnostics<T>,
        S
    }

    impl<D, S> PushOp<Ident<String>, S> for RelocContext<(), Downstream<Expr<S>, (), D>>
    where
        S: Clone,
    {
        fn push_op(&mut self, ident: Ident<String>, span: S) {
            use crate::model::{Atom, ExprItem};
            self.builder.backend.0.push(ExprItem {
                op: Atom::Name(ident).into(),
                op_span: span.clone(),
                expr_span: span,
            })
        }
    }

    impl<T, S> PushOp<Ident<String>, S> for RelocContext<MockDiagnostics<T>, Expr<S>>
    where
        T: From<DiagnosticsEvent<S>>,
        S: Clone,
    {
        fn push_op(&mut self, ident: Ident<String>, span: S) {
            use crate::model::{Atom, ExprItem};
            self.builder.0.push(ExprItem {
                op: Atom::Name(ident).into(),
                op_span: span.clone(),
                expr_span: span,
            })
        }
    }

    impl<T, S> PartialBackend<S> for MockSession<T, S>
    where
        T: From<BackendEvent<Expr<S>>>,
        S: Clone + Merge,
    {
        type Value = Expr<S>;

        fn emit_item(&mut self, item: Item<Self::Value>) {
            self.log.push(BackendEvent::EmitItem(item))
        }

        fn reserve(&mut self, bytes: Self::Value) {
            self.log.push(BackendEvent::Reserve(bytes))
        }

        fn set_origin(&mut self, origin: Self::Value) {
            self.log.push(BackendEvent::SetOrigin(origin))
        }
    }

    impl<T, S> StartSection<Ident<String>, S> for MockSession<T, S>
    where
        T: From<BackendEvent<Expr<S>>>,
        S: Clone + Merge,
    {
        fn start_section(&mut self, name: (Ident<String>, S)) {
            self.log.push(BackendEvent::StartSection((0, name.1)))
        }
    }

    pub(in crate::analysis) type MockBuilder<T, S> =
        RelocContext<(), Downstream<Expr<S>, (), MockDiagnostics<T>>>;

    impl<T, S> MockBuilder<T, S> {
        pub fn with_log(log: Log<T>) -> Self {
            Self {
                parent: (),
                builder: Downstream {
                    backend: Default::default(),
                    names: (),
                    diagnostics: MockDiagnostics::new(log),
                },
            }
        }
    }

    impl<T, S: Clone> Finish<S> for MockBuilder<T, S> {
        type Parent = MockDiagnostics<T>;
        type Value = Expr<S>;

        fn finish(self) -> (Self::Parent, Self::Value) {
            (self.builder.diagnostics, self.builder.backend)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::backend::BackendEvent;
    use crate::analysis::resolve::{BasicNameTable, NameTableEvent};
    use crate::analysis::semantics::AnalyzerEvent;
    use crate::analysis::syntax::{Command, Directive, Mnemonic, Token};
    use crate::analysis::{Literal, MockCodebase};
    use crate::diag::DiagnosticsEvent;
    use crate::log::*;
    use crate::model::{Atom, BinOp, Instruction, Nullary, Width};

    use std::fmt::Debug;
    use std::iter;

    type Expr<S> = crate::model::Expr<LocationCounter, usize, S>;

    #[test]
    fn emit_instruction_item() {
        let item = Item::Instruction(Instruction::Nullary(Nullary::Nop));
        let log =
            Fixture::<()>::default().log_session(|mut session| session.emit_item(item.clone()));
        assert_eq!(log, [BackendEvent::EmitItem(item).into()]);
    }

    #[test]
    fn define_label() {
        let label = "label";
        let log = Fixture::default().log_session(|session| {
            let mut builder = session.define_symbol(label.into(), ());
            builder.push_op(LocationCounter, ());
            builder.finish_fn_def();
        });
        assert_eq!(
            log,
            [
                NameTableEvent::StartScope(label.into()).into(),
                BackendEvent::DefineSymbol((0, ()), LocationCounter.into()).into()
            ]
        );
    }

    #[test]
    fn start_section() {
        let name: Ident<_> = "my_section".into();
        let log =
            Fixture::default().log_session(|mut session| session.start_section((name.clone(), ())));
        assert_eq!(log, [BackendEvent::StartSection((0, ())).into()])
    }

    #[test]
    fn look_up_section_name_after_definition() {
        let ident: Ident<_> = "my_section".into();
        let log = Fixture::default().log_session(|mut session| {
            session.start_section((ident.clone(), ()));
            let mut builder = session.build_value();
            builder.push_op(ident, ());
            let (s, value) = Finish::finish(builder);
            let item = Item::Data(value, Width::Word);
            session = s;
            session.emit_item(item)
        });
        assert_eq!(
            log,
            [
                BackendEvent::StartSection((0, ())).into(),
                BackendEvent::EmitItem(Item::Data(Atom::Name(0).into(), Width::Word)).into()
            ]
        )
    }

    #[test]
    fn include_source_file() {
        let path = "my_file.s";
        let tokens = vec![(Ok(Token::Command(Command::Mnemonic(Mnemonic::Nop))), ())];
        let log = Fixture::new(|fixture| fixture.codebase.set_file(path, tokens.clone()))
            .log_session(|session| session.analyze_file(path.into()).0.unwrap());
        assert_eq!(log, [AnalyzerEvent::AnalyzeTokenSeq(tokens).into()]);
    }

    #[test]
    fn define_and_call_macro() {
        let name = "my_macro";
        let tokens = vec![Token::Command(Command::Mnemonic(Mnemonic::Nop))];
        let spans: Vec<_> = iter::repeat(()).take(tokens.len()).collect();
        let log = Fixture::default().log_session(|mut session| {
            session.define_macro(
                (name.into(), ()),
                (vec![], vec![]),
                (tokens.clone(), spans.clone()),
            );
            session.call_macro((name.into(), ()), vec![]);
        });
        assert_eq!(
            log,
            [AnalyzerEvent::AnalyzeTokenSeq(
                tokens.into_iter().map(|token| (Ok(token), ())).collect()
            )
            .into()]
        );
    }

    #[test]
    fn define_and_call_macro_with_param() {
        let db = Token::Command(Command::Directive(Directive::Db));
        let arg = Token::Literal(Literal::Number(0x42));
        let literal0 = Token::Literal(Literal::Number(0));
        let name = "my_db";
        let param = "x";
        let log = Fixture::default().log_session(|mut session| {
            session.define_macro(
                (name.into(), ()),
                (vec![param.into()], vec![()]),
                (
                    vec![db.clone(), Token::Ident(param.into()), literal0.clone()],
                    vec![(), (), ()],
                ),
            );
            session.call_macro((name.into(), ()), vec![vec![(arg.clone(), ())]]);
        });
        assert_eq!(
            log,
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
    fn define_and_call_macro_with_label() {
        let nop = Token::Command(Command::Mnemonic(Mnemonic::Nop));
        let label = "label";
        let name = "my_macro";
        let param = "x";
        let log = Fixture::default().log_session(|mut session| {
            session.define_macro(
                (name.into(), ()),
                (vec![param.into()], vec![()]),
                (vec![Token::Label(param.into()), nop.clone()], vec![(), ()]),
            );
            session.call_macro(
                (name.into(), ()),
                vec![vec![(Token::Ident(label.into()), ())]],
            );
        });
        assert_eq!(
            log,
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
    fn reserve_bytes() {
        let bytes = 10;
        let log = Fixture::default().log_session(|mut session| session.reserve(bytes.into()));
        assert_eq!(log, [BackendEvent::Reserve(bytes.into()).into()])
    }

    #[test]
    fn diagnose_undefined_macro() {
        let name = "my_macro";
        let span = name;
        let log = Fixture::<MockSpan<_>>::default().log_session(|session| {
            session.call_macro((name.into(), span.into()), vec![]);
        });
        assert_eq!(
            log,
            [DiagnosticsEvent::EmitDiag(
                Message::UndefinedMacro { name: span.into() }
                    .at(span.into())
                    .into()
            )
            .into()]
        );
    }

    impl Default for MockSpan<&'static str> {
        fn default() -> Self {
            unreachable!()
        }
    }

    #[test]
    fn build_value_from_number() {
        Fixture::default().log_session(|session| {
            let mut builder = session.build_value();
            builder.push_op(42, ());
            let (_, value) = builder.finish();
            assert_eq!(value, 42.into())
        });
    }

    #[test]
    fn apply_operator_on_two_values() {
        Fixture::default().log_session(|session| {
            let mut builder = session.build_value();
            builder.push_op(42, ());
            builder.push_op(Ident::from("ident"), ());
            builder.push_op(BinOp::Multiplication, ());
            let (_, value) = builder.finish();
            assert_eq!(
                value,
                Expr::from_items(&[
                    42.into(),
                    Atom::Name(0).into(),
                    BinOp::Multiplication.into()
                ])
            )
        });
    }

    #[test]
    fn diagnose_macro_name_in_expr() {
        let ident = "my_macro";
        let log = Fixture::<MockSpan<_>>::default().log_session(|mut session| {
            session.define_macro(
                (ident.into(), "m".into()),
                (vec![], vec![]),
                (vec![], vec![]),
            );
            let mut builder = session.build_value();
            builder.push_op(Ident::from(ident), "ident".into());
            let (mut session, value) = builder.finish();
            session.emit_item(Item::Data(value, Width::Byte))
        });
        assert_eq!(
            log,
            [
                DiagnosticsEvent::EmitDiag(Message::MacroNameInExpr.at("ident".into()).into())
                    .into(),
                BackendEvent::EmitItem(Item::Data(
                    Expr::from_atom(Atom::Name(0), "ident".into()),
                    Width::Byte
                ))
                .into(),
            ]
        )
    }

    type MockAnalyzer<S> = crate::analysis::semantics::MockAnalyzer<Event<S>>;
    type MockBackend<S> = crate::analysis::backend::MockBackend<Event<S>>;
    type MockDiagnosticsSystem<S> = crate::diag::MockDiagnosticsSystem<Event<S>, S>;
    type MockNameTable<S> = crate::analysis::resolve::MockNameTable<
        BasicNameTable<usize, MacroEntry<String, MockDiagnosticsSystem<S>>>,
        Event<S>,
    >;
    type TestSession<'a, S> = CompositeSession<
        'a,
        'a,
        MockCodebase<S>,
        MockAnalyzer<S>,
        MockBackend<S>,
        MockNameTable<S>,
        MockDiagnosticsSystem<S>,
    >;

    #[derive(Debug, PartialEq)]
    enum Event<S: Clone> {
        Frontend(AnalyzerEvent<S>),
        Backend(BackendEvent<Expr<S>>),
        NameTable(NameTableEvent),
        Diagnostics(DiagnosticsEvent<S>),
    }

    impl<S: Clone> From<AnalyzerEvent<S>> for Event<S> {
        fn from(event: AnalyzerEvent<S>) -> Self {
            Event::Frontend(event)
        }
    }

    impl<S: Clone> From<BackendEvent<Expr<S>>> for Event<S> {
        fn from(event: BackendEvent<Expr<S>>) -> Self {
            Event::Backend(event)
        }
    }

    impl<S: Clone> From<NameTableEvent> for Event<S> {
        fn from(event: NameTableEvent) -> Self {
            Event::NameTable(event)
        }
    }

    impl<S: Clone> From<DiagnosticsEvent<S>> for Event<S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            Event::Diagnostics(event)
        }
    }

    struct Fixture<S: Clone + Default + Merge> {
        inner: InnerFixture<S>,
        log: Log<Event<S>>,
    }

    struct InnerFixture<S: Clone + Default + Merge> {
        codebase: MockCodebase<S>,
        analyzer: MockAnalyzer<S>,
        backend: Option<MockBackend<S>>,
        names: MockNameTable<S>,
        diagnostics: MockDiagnosticsSystem<S>,
    }

    impl<S: Clone + Default + Merge> Fixture<S> {
        fn new(f: impl FnOnce(&mut InnerFixture<S>)) -> Self {
            let log = Log::new();
            let mut fixture = Self {
                inner: InnerFixture {
                    codebase: MockCodebase::new(),
                    analyzer: MockAnalyzer::new(log.clone()),
                    backend: Some(MockBackend::new(log.clone())),
                    names: MockNameTable::new(BasicNameTable::new(), log.clone()),
                    diagnostics: MockDiagnosticsSystem::new(log.clone()),
                },
                log,
            };
            f(&mut fixture.inner);
            fixture
        }

        fn log_session(mut self, f: impl FnOnce(TestSession<S>)) -> Vec<Event<S>>
        where
            Event<S>: Debug,
        {
            f(CompositeSession::new(
                &mut self.inner.codebase,
                &mut self.inner.analyzer,
                self.inner.backend.take().unwrap(),
                &mut self.inner.names,
                &mut self.inner.diagnostics,
            ));
            drop(self.inner);
            self.log.into_inner()
        }
    }

    impl<S: Clone + Default + Merge> Default for Fixture<S> {
        fn default() -> Self {
            Self::new(|_| {})
        }
    }
}
