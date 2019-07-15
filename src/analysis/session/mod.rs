pub use super::backend::ValueBuilder;

use self::builder::Builder;
use self::expand::{DefineMacro, Expand, MacroId};

use super::backend::*;
use super::resolve::{Ident, NameTable, ResolvedIdent, StartScope};
use super::{Command, Lex, LexItem, Literal, SemanticToken, StringSource, TokenSeq};

use crate::codebase::CodebaseError;
use crate::diag::span::{AddMacroDef, SpanSource};
use crate::diag::*;
use crate::model::Item;

use std::ops::DerefMut;

#[cfg(test)]
pub(crate) use self::mock::*;

mod builder;
mod expand;

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
        body: TokenSeq<Self::StringRef, Self::Span>,
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
    Self: AllocName<S>,
    Self: PartialBackend<S>,
    Self: StartSection<<Self as AllocName<S>>::Name, S>,
    Self: StartScope<Ident<R>>,
    Self: NameTable<Ident<R>, BackendEntry = <Self as AllocName<S>>::Name>,
    Self: Diagnostics<S>,
{
    type FnBuilder: ValueBuilder<Self::Name, S>
        + AllocName<S, Name = Self::Name>
        + NameTable<Ident<R>, BackendEntry = Self::Name>
        + FinishFnDef<Return = Self>
        + Diagnostics<S>;
    type GeneralBuilder: ValueBuilder<Self::Name, S>
        + AllocName<S, Name = Self::Name>
        + NameTable<Ident<R>, BackendEntry = Self::Name>
        + Finish<S, Parent = Self, Value = Self::Value>
        + Diagnostics<S>;

    fn build_value(self) -> Self::GeneralBuilder;
    fn define_symbol(self, name: Self::Name, span: S) -> Self::FnBuilder;
}

pub(super) type MacroArgs<I, S> = expand::MacroArgs<SemanticToken<I>, S>;
pub(super) type Params<R, S> = (Vec<Ident<R>>, Vec<S>);

type FullSession<'a, 'b, C, A, B, N, D> =
    CompositeSession<FullUpstream<'a, 'b, C, A, D>, B, &'a mut N, &'a mut D>;

type FullUpstream<'a, 'b, C, A, D> = Upstream<
    'a,
    'b,
    C,
    A,
    <C as StringSource>::StringRef,
    <D as AddMacroDef<<D as SpanSource>::Span>>::MacroDefHandle,
>;

pub(super) struct CompositeSession<U, B, N, D> {
    upstream: U,
    downstream: Downstream<B, N, D>,
}

pub(super) struct Upstream<'a, 'b, C, A, R, H> {
    codebase: &'a mut C,
    analyzer: &'b mut A,
    macros: MacroTable<R, H>,
}

type MacroTable<R, H> = self::expand::MacroTable<Ident<R>, Literal<R>, Command, H>;

pub(super) struct Downstream<B, N, D> {
    backend: B,
    names: N,
    diagnostics: D,
}

impl<B: AllocName<S>, N, D, S: Clone> AllocName<S> for Downstream<B, N, D> {
    type Name = B::Name;

    fn alloc_name(&mut self, span: S) -> Self::Name {
        self.backend.alloc_name(span)
    }
}

impl<B, N, D, I> NameTable<I> for Downstream<B, N, D>
where
    N: DerefMut,
    N::Target: NameTable<I>,
{
    type BackendEntry = <N::Target as NameTable<I>>::BackendEntry;
    type MacroEntry = <N::Target as NameTable<I>>::MacroEntry;

    fn get(&self, ident: &I) -> Option<ResolvedIdent<Self::BackendEntry, Self::MacroEntry>> {
        self.names.get(ident)
    }

    fn insert(&mut self, ident: I, entry: ResolvedIdent<Self::BackendEntry, Self::MacroEntry>) {
        self.names.insert(ident, entry)
    }
}

impl<B: PushOp<T, S>, N, D, T, S: Clone> PushOp<T, S> for Downstream<B, N, D> {
    fn push_op(&mut self, op: T, span: S) {
        self.backend.push_op(op, span)
    }
}

impl<'a, 'b, C, A, B, N, D> FullSession<'a, 'b, C, A, B, N, D>
where
    C: StringSource,
    D: DiagnosticsSystem,
{
    pub fn new(
        codebase: &'a mut C,
        analyzer: &'b mut A,
        backend: B,
        names: &'a mut N,
        diagnostics: &'a mut D,
    ) -> Self {
        CompositeSession {
            upstream: Upstream {
                codebase,
                analyzer,
                macros: Vec::new(),
            },
            downstream: Downstream {
                backend,
                names,
                diagnostics,
            },
        }
    }
}

impl<B, N, D> Downstream<B, N, D> {
    fn replace_backend<T>(self, f: impl FnOnce(B) -> T) -> Downstream<T, N, D> {
        Downstream {
            backend: f(self.backend),
            names: self.names,
            diagnostics: self.diagnostics,
        }
    }
}

pub(super) struct PartialSession<'a, C, B, N, D>
where
    C: StringSource,
    D: DiagnosticsSystem,
{
    codebase: &'a mut C,
    macros: MacroTable<C::StringRef, D::MacroDefHandle>,
    downstream: Downstream<B, &'a mut N, &'a mut D>,
}

macro_rules! partial {
    ($session:expr) => {
        PartialSession {
            codebase: $session.upstream.codebase,
            macros: $session.upstream.macros,
            downstream: Downstream {
                backend: $session.downstream.backend,
                names: $session.downstream.names,
                diagnostics: $session.downstream.diagnostics,
            },
        }
    };
}

pub(super) trait IntoSession<'b, A>
where
    Self: Sized,
    A: ?Sized,
{
    type Session: Session + Into<Self>;

    fn into_session(self, analyzer: &'b mut A) -> Self::Session;
}

impl<'a, 'b, C, A: 'b, B, N, D> IntoSession<'b, A> for PartialSession<'a, C, B, N, D>
where
    C: StringSource,
    D: DiagnosticsSystem,
    FullSession<'a, 'b, C, A, B, N, D>: Session + Into<Self>,
{
    type Session = FullSession<'a, 'b, C, A, B, N, D>;

    fn into_session(self, analyzer: &'b mut A) -> Self::Session {
        CompositeSession {
            upstream: Upstream {
                codebase: self.codebase,
                analyzer,
                macros: self.macros,
            },
            downstream: self.downstream,
        }
    }
}

impl<'a, 'b, C, A, B, N, D> From<FullSession<'a, 'b, C, A, B, N, D>>
    for PartialSession<'a, C, B, N, D>
where
    C: StringSource,
    D: DiagnosticsSystem,
{
    fn from(session: FullSession<'a, 'b, C, A, B, N, D>) -> Self {
        partial!(session)
    }
}

impl<U, B, N, D> SpanSource for CompositeSession<U, B, N, D>
where
    D: DerefMut,
    D::Target: SpanSource,
{
    type Span = <D::Target as SpanSource>::Span;
}

impl<'a, 'b, C, A, B, N, D> StringSource for FullSession<'a, 'b, C, A, B, N, D>
where
    C: StringSource,
    D: DiagnosticsSystem,
{
    type StringRef = C::StringRef;
}

impl<U, B: Backend<S>, N, D, S: Clone> PartialBackend<S> for CompositeSession<U, B, N, D> {
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

pub(super) trait Analyze<R: Clone + Eq, S: Clone> {
    fn analyze_token_seq<'b, I, P>(&'b mut self, tokens: I, partial: P) -> P
    where
        I: IntoIterator<Item = LexItem<R, S>>,
        P: IntoSession<'b, Self>,
        P::Session: StringSource<StringRef = R> + SpanSource<Span = S>;
}

impl<'a, 'b, C, A, B, N, D> Session for FullSession<'a, 'b, C, A, B, N, D>
where
    C: Lex<D>,
    A: Analyze<C::StringRef, D::Span>,
    B: Backend<D::Span>,
    N: NameTable<Ident<C::StringRef>, BackendEntry = B::Name, MacroEntry = MacroId>
        + StartScope<Ident<C::StringRef>>,
    D: DiagnosticsSystem,
{
    fn analyze_file(mut self, path: Self::StringRef) -> (Result<(), CodebaseError>, Self) {
        let tokens = match self
            .upstream
            .codebase
            .lex_file(path, self.downstream.diagnostics)
        {
            Ok(tokens) => tokens,
            Err(error) => return (Err(error), self),
        };
        let PartialSession {
            macros,
            downstream: Downstream { backend, .. },
            ..
        } = self
            .upstream
            .analyzer
            .analyze_token_seq(tokens, partial!(self));
        self.upstream.macros = macros;
        self.downstream.backend = backend;
        (Ok(()), self)
    }

    fn define_macro(
        &mut self,
        name: (Ident<Self::StringRef>, Self::Span),
        params: Params<Self::StringRef, Self::Span>,
        body: TokenSeq<Self::StringRef, Self::Span>,
    ) {
        let id = self.upstream.macros.define_macro(
            name.clone(),
            params,
            body,
            self.downstream.diagnostics,
        );
        self.downstream
            .names
            .insert(name.0, ResolvedIdent::Macro(id))
    }

    fn call_macro(
        mut self,
        name: (Ident<Self::StringRef>, Self::Span),
        args: MacroArgs<Self::StringRef, Self::Span>,
    ) -> Self {
        let stripped = self.downstream.diagnostics.strip_span(&name.1);
        let expansion = match self.downstream.names.get(&name.0) {
            Some(ResolvedIdent::Macro(MacroId(id))) => {
                let def = &self.upstream.macros[id];
                Ok(def.expand(name.1, args, self.downstream.diagnostics))
            }
            Some(ResolvedIdent::Backend(_)) => {
                Err(Message::CannotUseSymbolNameAsMacroName { name: stripped }.at(name.1))
            }
            None => Err(Message::UndefinedMacro { name: stripped }.at(name.1)),
        }
        .map_err(|diag| self.downstream.diagnostics.emit_diag(diag))
        .ok();
        if let Some(expansion) = expansion {
            let PartialSession {
                macros,
                downstream: Downstream { backend, .. },
                ..
            } = self
                .upstream
                .analyzer
                .analyze_token_seq(expansion.map(|(t, s)| (Ok(t), s)), partial!(self));
            self.upstream.macros = macros;
            self.downstream.backend = backend
        }
        self
    }
}

impl<U, B, N, D, R, S> BasicSession<R, S> for CompositeSession<U, B, N, D>
where
    B: Backend<S>,
    N: DerefMut,
    N::Target: NameTable<Ident<R>, BackendEntry = B::Name> + StartScope<Ident<R>>,
    D: DerefMut,
    D::Target: Diagnostics<S>,
    S: Clone,
    Self: Diagnostics<S>,
    Builder<U, B::SymbolBuilder, N, D>: Diagnostics<S>,
    Builder<U, B::ImmediateBuilder, N, D>: Diagnostics<S>,
{
    type FnBuilder = Builder<U, B::SymbolBuilder, N, D>;
    type GeneralBuilder = Builder<U, B::ImmediateBuilder, N, D>;

    fn build_value(self) -> Self::GeneralBuilder {
        RelocContext {
            parent: self.upstream,
            builder: self.downstream.replace_backend(Backend::build_immediate),
        }
    }

    fn define_symbol(self, name: B::Name, span: S) -> Self::FnBuilder {
        RelocContext {
            parent: self.upstream,
            builder: self
                .downstream
                .replace_backend(|backend| backend.define_symbol(name, span)),
        }
    }
}

impl<U, B: AllocName<S>, N, D, S: Clone> AllocName<S> for CompositeSession<U, B, N, D> {
    type Name = B::Name;

    fn alloc_name(&mut self, span: S) -> Self::Name {
        self.downstream.backend.alloc_name(span)
    }
}

impl<U, B, N, D, R> NameTable<Ident<R>> for CompositeSession<U, B, N, D>
where
    N: DerefMut,
    N::Target: NameTable<Ident<R>>,
{
    type BackendEntry = <N::Target as NameTable<Ident<R>>>::BackendEntry;
    type MacroEntry = <N::Target as NameTable<Ident<R>>>::MacroEntry;

    fn get(&self, ident: &Ident<R>) -> Option<ResolvedIdent<Self::BackendEntry, Self::MacroEntry>> {
        self.downstream.get(ident)
    }

    fn insert(
        &mut self,
        ident: Ident<R>,
        entry: ResolvedIdent<Self::BackendEntry, Self::MacroEntry>,
    ) {
        self.downstream.insert(ident, entry)
    }
}

delegate_diagnostics! {
    {'a, U, B, N, D: Diagnostics<S>, S: Clone},
    CompositeSession<U, B, N, &'a mut D>,
    {downstream.diagnostics},
    D,
    S
}

impl<U, B, N, D, R> StartScope<Ident<R>> for CompositeSession<U, B, N, D>
where
    N: DerefMut,
    N::Target: StartScope<Ident<R>>,
{
    fn start_scope(&mut self, ident: &Ident<R>) {
        self.downstream.names.start_scope(ident)
    }
}

impl<U, B, N, D, S> StartSection<B::Name, S> for CompositeSession<U, B, N, D>
where
    B: Backend<S>,
    S: Clone,
{
    fn start_section(&mut self, id: (B::Name, S)) {
        self.downstream.backend.start_section(id)
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::analysis::backend::BackendEvent;
    use crate::analysis::resolve::{MockNameTable, NameTableEvent};
    use crate::diag::{DiagnosticsEvent, MockDiagnostics};
    use crate::log::Log;
    use crate::model::Atom;

    use std::marker::PhantomData;

    type Expr<N, S> = crate::model::Expr<Atom<LocationCounter, N>, S>;

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

    pub(in crate::analysis) type MockSession<A, N, T, S> = CompositeSession<
        MockUpstream<T, S>,
        MockBackend<A, T>,
        Box<MockNameTable<N, T>>,
        Box<MockDiagnostics<T, S>>,
    >;

    pub(in crate::analysis) struct MockUpstream<T, S> {
        log: Log<T>,
        error: Option<CodebaseError>,
        _span: PhantomData<S>,
    }

    impl<A, N, T, S> MockSession<A, N, T, S> {
        pub fn with_name_table(alloc: A, names: N, log: Log<T>) -> Self {
            Self {
                downstream: Downstream {
                    backend: MockBackend::new(alloc, log.clone()),
                    names: Box::new(MockNameTable::new(names, log.clone())),
                    diagnostics: Box::new(MockDiagnostics::new(log.clone())),
                },
                upstream: MockUpstream {
                    log,
                    error: None,
                    _span: PhantomData,
                },
            }
        }

        pub fn fail(&mut self, error: CodebaseError) {
            self.upstream.error = Some(error)
        }
    }

    delegate_diagnostics! {
        {A, N, T: From<DiagnosticsEvent<S>>, S: Merge},
        MockSession<A, N, T, S>,
        {downstream.diagnostics},
        MockDiagnostics<T, S>,
        S
    }

    impl<A, N, T, S> StringSource for MockSession<A, N, T, S> {
        type StringRef = String;
    }

    impl<A, N, T, S> Session for MockSession<A, N, T, S>
    where
        A: AllocName<S>,
        N: NameTable<Ident<String>, BackendEntry = A::Name>,
        T: From<SessionEvent>,
        T: From<BackendEvent<A::Name, Expr<A::Name, S>>>,
        T: From<DiagnosticsEvent<S>>,
        T: From<NameTableEvent<N::BackendEntry, N::MacroEntry>>,
        S: Clone + Merge,
    {
        fn analyze_file(mut self, path: String) -> (Result<(), CodebaseError>, Self) {
            self.upstream.log.push(SessionEvent::AnalyzeFile(path));
            (self.upstream.error.take().map_or(Ok(()), Err), self)
        }

        fn define_macro(
            &mut self,
            name: (Ident<Self::StringRef>, Self::Span),
            params: (Vec<Ident<Self::StringRef>>, Vec<Self::Span>),
            body: (Vec<SemanticToken<Self::StringRef>>, Vec<Self::Span>),
        ) {
            self.upstream
                .log
                .push(SessionEvent::DefineMacro(name.0, params.0, body.0))
        }

        fn call_macro(
            self,
            name: (Ident<Self::StringRef>, Self::Span),
            (args, _): MacroArgs<Self::StringRef, Self::Span>,
        ) -> Self {
            self.upstream
                .log
                .push(SessionEvent::InvokeMacro(name.0, args));
            self
        }
    }

    pub(in crate::analysis) type MockBuilder<U, A, N, T, S> = RelocContext<
        U,
        Downstream<
            RelocContext<MockBackend<A, T>, Expr<<A as AllocName<S>>::Name, S>>,
            Box<MockNameTable<N, T>>,
            Box<MockDiagnostics<T, S>>,
        >,
    >;

    impl<A, N, T, S> MockBuilder<(), A, N, T, S>
    where
        A: AllocName<S>,
        N: NameTable<Ident<String>>,
        T: From<BackendEvent<A::Name, Expr<A::Name, S>>>,
        S: Clone,
    {
        pub fn from_components(alloc: A, names: N, log: Log<T>) -> Self {
            Self {
                parent: (),
                builder: Downstream {
                    backend: MockBackend::new(alloc, log.clone()).build_immediate(),
                    names: Box::new(MockNameTable::new(names, log.clone())),
                    diagnostics: Box::new(MockDiagnostics::new(log)),
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::backend::{BackendEvent, SerialIdAllocator};
    use crate::analysis::resolve::{BasicNameTable, NameTableEvent};
    use crate::analysis::semantics::AnalyzerEvent;
    use crate::analysis::syntax::{Command, Directive, Mnemonic, Token};
    use crate::analysis::{Literal, MockCodebase};
    use crate::diag::DiagnosticsEvent;
    use crate::log::*;
    use crate::model::{Atom, BinOp, Instruction, Nullary};

    use std::fmt::Debug;
    use std::iter;

    type Expr<S> = crate::model::Expr<Atom<LocationCounter, usize>, S>;

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
        let log = Fixture::default().log_session(|mut session| {
            let id = session.alloc_name(());
            session.insert(label.into(), ResolvedIdent::Backend(id.clone()));
            let mut builder = session.define_symbol(id, ());
            builder.push_op(LocationCounter, ());
            builder.finish_fn_def();
        });
        let id = 0;
        assert_eq!(
            log,
            [
                NameTableEvent::Insert(label.into(), ResolvedIdent::Backend(id)).into(),
                BackendEvent::DefineSymbol((id, ()), LocationCounter.into()).into()
            ]
        );
    }

    #[test]
    fn start_section() {
        let name: Ident<_> = "my_section".into();
        let log = Fixture::default().log_session(|mut session| {
            let id = session.alloc_name(());
            session.insert(name.clone(), ResolvedIdent::Backend(id));
            session.start_section((id, ()))
        });
        let id = 0;
        assert_eq!(
            log,
            [
                NameTableEvent::Insert(name, ResolvedIdent::Backend(id)).into(),
                BackendEvent::StartSection((id, ())).into()
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
            session.call_macro((name.into(), ()), (vec![], vec![]));
        });
        assert_eq!(
            log,
            [
                NameTableEvent::Insert(name.into(), ResolvedIdent::Macro(MacroId(0))).into(),
                AnalyzerEvent::AnalyzeTokenSeq(
                    tokens.into_iter().map(|token| (Ok(token), ())).collect()
                )
                .into()
            ]
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
            session.call_macro((name.into(), ()), (vec![vec![arg.clone()]], vec![vec![()]]));
        });
        assert_eq!(
            log,
            [
                NameTableEvent::Insert(name.into(), ResolvedIdent::Macro(MacroId(0))).into(),
                AnalyzerEvent::AnalyzeTokenSeq(
                    vec![db, arg, literal0]
                        .into_iter()
                        .map(|token| (Ok(token), ()))
                        .collect()
                )
                .into()
            ]
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
                (vec![vec![Token::Ident(label.into())]], vec![vec![()]]),
            );
        });
        assert_eq!(
            log,
            [
                NameTableEvent::Insert(name.into(), ResolvedIdent::Macro(MacroId(0))).into(),
                AnalyzerEvent::AnalyzeTokenSeq(
                    vec![Token::Label(label.into()), nop]
                        .into_iter()
                        .map(|token| (Ok(token), ()))
                        .collect()
                )
                .into()
            ]
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
            session.call_macro((name.into(), span.into()), (vec![], vec![]));
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
            builder.push_op(Name(0), ());
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
    fn diagnose_symbol_name_in_macro_call() {
        let name = "symbol";
        let as_macro = MockSpan::from("as_macro");
        let log = Fixture::<MockSpan<_>>::default().log_session(|mut session| {
            let id = session.alloc_name(MockSpan::from("as_symbol"));
            session.insert(Ident::from(name), ResolvedIdent::Backend(id));
            session.call_macro((name.into(), as_macro.clone()), (vec![], vec![]));
        });
        assert_eq!(
            log,
            [
                NameTableEvent::Insert(name.into(), ResolvedIdent::Backend(0)).into(),
                DiagnosticsEvent::EmitDiag(
                    Message::CannotUseSymbolNameAsMacroName {
                        name: as_macro.clone()
                    }
                    .at(as_macro)
                    .into()
                )
                .into()
            ]
        )
    }

    type MockAnalyzer<S> = crate::analysis::semantics::MockAnalyzer<Event<S>>;
    type MockBackend<S> = crate::analysis::backend::MockBackend<SerialIdAllocator, Event<S>>;
    type MockDiagnosticsSystem<S> = crate::diag::MockDiagnosticsSystem<Event<S>, S>;
    type MockNameTable<S> =
        crate::analysis::resolve::MockNameTable<BasicNameTable<usize, MacroId>, Event<S>>;
    type TestSession<'a, S> = FullSession<
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
        Backend(BackendEvent<usize, Expr<S>>),
        NameTable(NameTableEvent<usize, MacroId>),
        Diagnostics(DiagnosticsEvent<S>),
    }

    impl<S: Clone> From<AnalyzerEvent<S>> for Event<S> {
        fn from(event: AnalyzerEvent<S>) -> Self {
            Event::Frontend(event)
        }
    }

    impl<S: Clone> From<BackendEvent<usize, Expr<S>>> for Event<S> {
        fn from(event: BackendEvent<usize, Expr<S>>) -> Self {
            Event::Backend(event)
        }
    }

    impl<S: Clone> From<NameTableEvent<usize, MacroId>> for Event<S> {
        fn from(event: NameTableEvent<usize, MacroId>) -> Self {
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
                    backend: Some(MockBackend::new(SerialIdAllocator::new(), log.clone())),
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
