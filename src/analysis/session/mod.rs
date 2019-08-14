use self::expand::{DefineMacro, Expand, MacroId, MacroTable};

use super::resolve::{NameTable, ResolvedIdent, StartScope};
use super::syntax::actions::TokenStreamActions;
use super::syntax::parser::ParserFactory;
use super::syntax::{LexError, ParseTokenStream};
use super::{IdentSource, Lex, Literal, SemanticToken, StringSource, TokenSeq};

use crate::codebase::CodebaseError;
use crate::diag::span::{AddMacroDef, SpanSource};
use crate::diag::*;
use crate::object::builder::*;

use std::ops::DerefMut;

#[cfg(test)]
pub(crate) use self::mock::*;

mod builder;
mod expand;

pub(super) trait Session
where
    Self: IdentSource + SpanSource + StringSource,
    Self: PartialSession<<Self as IdentSource>::Ident, <Self as SpanSource>::Span>,
{
    fn analyze_file<A: IntoSemanticActions<Self>>(
        self,
        path: Self::StringRef,
        actions: A,
    ) -> (Result<(), CodebaseError>, A::SemanticActions)
    where
        A::SemanticActions: TokenStreamActions<Self::Ident, Literal<Self::StringRef>, Self::Span>;

    fn define_macro(
        &mut self,
        name_span: Self::Span,
        params: Params<Self::Ident, Self::Span>,
        body: TokenSeq<Self::Ident, Self::StringRef, Self::Span>,
    ) -> Self::MacroEntry;

    fn call_macro<A: IntoSemanticActions<Self>>(
        self,
        name: (Self::MacroEntry, Self::Span),
        args: MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
        actions: A,
    ) -> A::SemanticActions
    where
        A::SemanticActions: TokenStreamActions<Self::Ident, Literal<Self::StringRef>, Self::Span>;
}

pub(super) trait PartialSession<I, S: Clone>
where
    Self: Sized,
    Self: AllocName<S>,
    Self: PartialBackend<S>,
    Self: StartSection<<Self as AllocName<S>>::Name, S>,
    Self: StartScope<I>,
    Self: NameTable<I, BackendEntry = <Self as AllocName<S>>::Name>,
    Self: Diagnostics<S>,
{
    type ConstBuilder: ValueBuilder<Self::Name, S>
        + AllocName<S, Name = Self::Name>
        + NameTable<I, BackendEntry = Self::Name>
        + Finish<Parent = Self, Value = Self::Value>
        + Diagnostics<S>;
    type SymbolBuilder: ValueBuilder<Self::Name, S>
        + AllocName<S, Name = Self::Name>
        + NameTable<I, BackendEntry = Self::Name>
        + Finish<Parent = Self, Value = ()>
        + Diagnostics<S>;

    fn build_const(self) -> Self::ConstBuilder;
    fn define_symbol(self, name: Self::Name, span: S) -> Self::SymbolBuilder;
}

pub(super) trait IntoSemanticActions<S: Session> {
    type SemanticActions;

    fn into_semantic_actions(self, session: S) -> Self::SemanticActions;
}

pub(super) type MacroArgs<I, R, S> = expand::MacroArgs<SemanticToken<I, R>, S>;
pub(super) type Params<I, S> = (Vec<I>, Vec<S>);

pub(super) struct SessionComponents<U, B, N, D> {
    upstream: U,
    backend: B,
    names: N,
    diagnostics: D,
}

pub(super) struct Upstream<C, P, I, R, H> {
    codebase: C,
    parser_factory: P,
    macros: MacroTable<I, Literal<R>, H>,
}

impl<'a, C, P, B, N, D>
    SessionComponents<
        Upstream<
            &'a mut C,
            &'a mut P,
            <C as IdentSource>::Ident,
            <C as StringSource>::StringRef,
            <D as AddMacroDef<<D as SpanSource>::Span>>::MacroDefHandle,
        >,
        B,
        &'a mut N,
        &'a mut D,
    >
where
    C: IdentSource + StringSource,
    D: DiagnosticsSystem,
{
    pub fn new(
        codebase: &'a mut C,
        parser_factory: &'a mut P,
        backend: B,
        names: &'a mut N,
        diagnostics: &'a mut D,
    ) -> Self {
        SessionComponents {
            upstream: Upstream {
                codebase,
                parser_factory,
                macros: Vec::new(),
            },
            backend,
            names,
            diagnostics,
        }
    }
}

impl<U, B, N, D> SessionComponents<U, B, N, D> {
    fn replace_backend<T>(self, f: impl FnOnce(B) -> T) -> SessionComponents<U, T, N, D> {
        SessionComponents {
            upstream: self.upstream,
            backend: f(self.backend),
            names: self.names,
            diagnostics: self.diagnostics,
        }
    }
}

impl<C, P, B, N, D> IdentSource
    for SessionComponents<
        Upstream<
            C,
            P,
            <C::Target as IdentSource>::Ident,
            <C::Target as StringSource>::StringRef,
            <D::Target as AddMacroDef<<D::Target as SpanSource>::Span>>::MacroDefHandle,
        >,
        B,
        N,
        D,
    >
where
    C: DerefMut,
    C::Target: IdentSource + StringSource,
    D: DerefMut,
    D::Target: DiagnosticsSystem,
{
    type Ident = <C::Target as IdentSource>::Ident;
}

impl<U, B, N, D> SpanSource for SessionComponents<U, B, N, D>
where
    D: DerefMut,
    D::Target: SpanSource,
{
    type Span = <D::Target as SpanSource>::Span;
}

impl<C, P, B, N, D> StringSource
    for SessionComponents<
        Upstream<
            C,
            P,
            <C::Target as IdentSource>::Ident,
            <C::Target as StringSource>::StringRef,
            <D::Target as AddMacroDef<<D::Target as SpanSource>::Span>>::MacroDefHandle,
        >,
        B,
        N,
        D,
    >
where
    C: DerefMut,
    C::Target: IdentSource + StringSource,
    D: DerefMut,
    D::Target: DiagnosticsSystem,
{
    type StringRef = <C::Target as StringSource>::StringRef;
}

impl<U, B: Backend<S>, N, D, S: Clone> PartialBackend<S> for SessionComponents<U, B, N, D> {
    type Value = B::Value;

    fn emit_item(&mut self, item: Item<Self::Value>) {
        self.backend.emit_item(item)
    }

    fn reserve(&mut self, bytes: Self::Value) {
        self.backend.reserve(bytes)
    }

    fn set_origin(&mut self, origin: Self::Value) {
        self.backend.set_origin(origin)
    }
}

impl<C, P, B, N, D> Session
    for SessionComponents<
        Upstream<
            C,
            P,
            <C::Target as IdentSource>::Ident,
            <C::Target as StringSource>::StringRef,
            <D::Target as AddMacroDef<<D::Target as SpanSource>::Span>>::MacroDefHandle,
        >,
        B,
        N,
        D,
    >
where
    C: DerefMut,
    C::Target: Lex<D::Target>,
    P: DerefMut,
    P::Target: ParserFactory<
        <Self as IdentSource>::Ident,
        Literal<<Self as StringSource>::StringRef>,
        LexError,
        <Self as SpanSource>::Span,
    >,
    B: Backend<<D::Target as SpanSource>::Span>,
    N: DerefMut,
    N::Target:
        NameTable<<C::Target as IdentSource>::Ident, BackendEntry = B::Name, MacroEntry = MacroId>
            + StartScope<<C::Target as IdentSource>::Ident>,
    D: DerefMut,
    D::Target: DiagnosticsSystem,
{
    fn analyze_file<A: IntoSemanticActions<Self>>(
        mut self,
        path: Self::StringRef,
        actions: A,
    ) -> (Result<(), CodebaseError>, A::SemanticActions)
    where
        A::SemanticActions: TokenStreamActions<Self::Ident, Literal<Self::StringRef>, Self::Span>,
    {
        let tokens = match self
            .upstream
            .codebase
            .lex_file(path, &mut *self.diagnostics)
        {
            Ok(tokens) => tokens,
            Err(error) => return (Err(error), actions.into_semantic_actions(self)),
        };
        let mut parser = self.upstream.parser_factory.mk_parser();
        let actions = actions.into_semantic_actions(self);
        (Ok(()), parser.parse_token_stream(tokens, actions))
    }

    fn define_macro(
        &mut self,
        name_span: Self::Span,
        params: Params<Self::Ident, Self::Span>,
        body: TokenSeq<Self::Ident, Self::StringRef, Self::Span>,
    ) -> Self::MacroEntry {
        self.upstream
            .macros
            .define_macro(name_span, params, body, &mut *self.diagnostics)
    }

    fn call_macro<A: IntoSemanticActions<Self>>(
        mut self,
        (MacroId(id), span): (Self::MacroEntry, Self::Span),
        args: MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
        actions: A,
    ) -> A::SemanticActions
    where
        A::SemanticActions: TokenStreamActions<Self::Ident, Literal<Self::StringRef>, Self::Span>,
    {
        let expansion = self.upstream.macros[id].expand(span, args, &mut *self.diagnostics);
        let mut parser = self.upstream.parser_factory.mk_parser();
        let actions = actions.into_semantic_actions(self);
        parser.parse_token_stream(expansion.map(|(t, s)| (Ok(t), s)), actions)
    }
}

impl<U, B, N, D, I, S> PartialSession<I, S> for SessionComponents<U, B, N, D>
where
    B: Backend<S>,
    N: DerefMut,
    N::Target: NameTable<I, BackendEntry = B::Name> + StartScope<I>,
    D: DerefMut,
    D::Target: Diagnostics<S>,
    S: Clone,
    Self: Diagnostics<S>,
{
    type ConstBuilder = SessionComponents<U, B::ConstBuilder, N, D>;
    type SymbolBuilder = SessionComponents<U, B::SymbolBuilder, N, D>;

    fn build_const(self) -> Self::ConstBuilder {
        self.replace_backend(Backend::build_const)
    }

    fn define_symbol(self, name: B::Name, span: S) -> Self::SymbolBuilder {
        self.replace_backend(|backend| backend.define_symbol(name, span))
    }
}

impl<U, B: AllocName<S>, N, D, S: Clone> AllocName<S> for SessionComponents<U, B, N, D> {
    type Name = B::Name;

    fn alloc_name(&mut self, span: S) -> Self::Name {
        self.backend.alloc_name(span)
    }
}

impl<U, B, N, D, I> NameTable<I> for SessionComponents<U, B, N, D>
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

delegate_diagnostics! {
    {'a, U, B, N, D: DerefMut, S: Clone},
    {D::Target: Diagnostics<S>},
    SessionComponents<U, B, N, D>,
    {diagnostics},
    D::Target,
    S
}

impl<U, B, N, D, I> StartScope<I> for SessionComponents<U, B, N, D>
where
    N: DerefMut,
    N::Target: StartScope<I>,
{
    fn start_scope(&mut self, ident: &I) {
        self.names.start_scope(ident)
    }
}

impl<U, B, N, D, S> StartSection<B::Name, S> for SessionComponents<U, B, N, D>
where
    B: Backend<S>,
    S: Clone,
{
    fn start_section(&mut self, id: (B::Name, S)) {
        self.backend.start_section(id)
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::analysis::resolve::{BasicNameTable, FakeNameTable, MockNameTable, NameTableEvent};
    use crate::diag::{DiagnosticsEvent, MockDiagnostics};
    use crate::log::Log;
    use crate::model::{Atom, LocationCounter};
    use crate::object::builder::mock::*;

    use std::marker::PhantomData;

    type Expr<N, S> = crate::model::Expr<Atom<LocationCounter, N>, S>;

    #[derive(Debug, PartialEq)]
    pub(crate) enum SessionEvent {
        AnalyzeFile(String),
        DefineMacro(Vec<String>, Vec<SemanticToken<String, String>>),
        InvokeMacro(MockMacroId, Vec<Vec<SemanticToken<String, String>>>),
    }

    pub(in crate::analysis) type MockSession<A, N, T, S> = SessionComponents<
        MockUpstream<T, S>,
        MockBackend<A, T>,
        Box<MockNameTable<N, T>>,
        Box<MockDiagnostics<T, S>>,
    >;

    pub(in crate::analysis) struct MockUpstream<T, S> {
        id_gen: SerialIdAllocator,
        log: Log<T>,
        error: Option<CodebaseError>,
        _span: PhantomData<S>,
    }

    impl<A, N, T, S> MockSession<A, N, T, S> {
        fn with_name_table(alloc: A, names: N, log: Log<T>) -> Self {
            Self {
                backend: MockBackend::new(alloc, log.clone()),
                names: Box::new(MockNameTable::new(names, log.clone())),
                diagnostics: Box::new(MockDiagnostics::new(log.clone())),
                upstream: MockUpstream {
                    id_gen: SerialIdAllocator::new(),
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

    impl<T, S> MockSession<SerialIdAllocator, BasicNameTable<usize, MockMacroId>, T, S> {
        pub fn with_log(log: Log<T>) -> Self {
            Self::with_name_table(SerialIdAllocator::new(), BasicNameTable::new(), log)
        }

        pub fn with_predefined_names<I>(log: Log<T>, entries: I) -> Self
        where
            I: IntoIterator<Item = (String, ResolvedIdent<usize, MockMacroId>)>,
        {
            let mut table = BasicNameTable::new();
            for (name, value) in entries {
                table.insert(name, value)
            }
            Self::with_name_table(SerialIdAllocator::new(), table, log)
        }
    }

    impl<T, S> MockSession<PanickingIdAllocator<String>, FakeNameTable, T, S> {
        pub fn without_name_resolution(log: Log<T>) -> Self {
            Self::with_name_table(PanickingIdAllocator::new(), FakeNameTable, log)
        }
    }

    impl<A, N, T, S> IdentSource for MockSession<A, N, T, S> {
        type Ident = String;
    }

    impl<A, N, T, S> StringSource for MockSession<A, N, T, S> {
        type StringRef = String;
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct MockMacroId(pub usize);

    impl<B, N, T, S> Session for MockSession<B, N, T, S>
    where
        B: AllocName<S>,
        N: NameTable<String, BackendEntry = B::Name, MacroEntry = MockMacroId>,
        T: From<SessionEvent>,
        T: From<BackendEvent<B::Name, Expr<B::Name, S>>>,
        T: From<DiagnosticsEvent<S>>,
        T: From<NameTableEvent<N::BackendEntry, N::MacroEntry>>,
        S: Clone + Merge,
    {
        fn analyze_file<A: IntoSemanticActions<Self>>(
            mut self,
            path: String,
            actions: A,
        ) -> (Result<(), CodebaseError>, A::SemanticActions) {
            self.upstream.log.push(SessionEvent::AnalyzeFile(path));
            (
                self.upstream.error.take().map_or(Ok(()), Err),
                actions.into_semantic_actions(self),
            )
        }

        fn define_macro(
            &mut self,
            _: Self::Span,
            (params, _): (Vec<Self::Ident>, Vec<Self::Span>),
            (body, _): TokenSeq<Self::Ident, Self::StringRef, Self::Span>,
        ) -> Self::MacroEntry {
            self.upstream
                .log
                .push(SessionEvent::DefineMacro(params, body));
            MockMacroId(self.upstream.id_gen.gen())
        }

        fn call_macro<A: IntoSemanticActions<Self>>(
            self,
            (id, _): (Self::MacroEntry, Self::Span),
            (args, _): MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
            actions: A,
        ) -> A::SemanticActions {
            self.upstream.log.push(SessionEvent::InvokeMacro(id, args));
            actions.into_semantic_actions(self)
        }
    }

    pub(in crate::analysis) type MockBuilder<A, N, T, S> = SessionComponents<
        (),
        RelocContext<MockBackend<A, T>, Expr<<A as AllocName<S>>::Name, S>>,
        Box<MockNameTable<N, T>>,
        Box<MockDiagnostics<T, S>>,
    >;

    impl<A, N, T, S> MockBuilder<A, N, T, S>
    where
        A: AllocName<S>,
        N: NameTable<String>,
        T: From<BackendEvent<A::Name, Expr<A::Name, S>>>,
        S: Clone,
    {
        fn from_components(alloc: A, names: N, log: Log<T>) -> Self {
            Self {
                upstream: (),
                backend: MockBackend::new(alloc, log.clone()).build_const(),
                names: Box::new(MockNameTable::new(names, log.clone())),
                diagnostics: Box::new(MockDiagnostics::new(log)),
            }
        }
    }

    impl<T, S> MockBuilder<SerialIdAllocator, BasicNameTable<usize, usize>, T, S>
    where
        T: From<BackendEvent<usize, Expr<usize, S>>>,
        S: Clone,
    {
        pub fn with_name_resolution(log: Log<T>) -> Self {
            Self::with_predefined_names(log, std::iter::empty())
        }

        pub fn with_predefined_names<I>(log: Log<T>, entries: I) -> Self
        where
            I: IntoIterator<Item = (String, ResolvedIdent<usize, usize>)>,
        {
            let mut table = BasicNameTable::new();
            for (name, value) in entries {
                table.insert(name, value)
            }
            Self::from_components(SerialIdAllocator::new(), table, log)
        }
    }

    impl<T, S> MockBuilder<PanickingIdAllocator<String>, FakeNameTable, T, S>
    where
        T: From<BackendEvent<String, Expr<String, S>>>,
        S: Clone,
    {
        pub fn without_name_resolution(log: Log<T>) -> Self {
            Self::from_components(PanickingIdAllocator::new(), FakeNameTable, log)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::MacroId;

    use crate::analysis::resolve::{BasicNameTable, NameTableEvent};
    use crate::analysis::syntax::actions::mock::{IdentKind, TokenStreamActionCollector};
    use crate::analysis::syntax::parser::mock::*;
    use crate::analysis::syntax::*;
    use crate::analysis::{Literal, MockCodebase};
    use crate::diag::DiagnosticsEvent;
    use crate::log::*;
    use crate::model::{Atom, BinOp, LocationCounter};
    use crate::object::builder::mock::{BackendEvent, SerialIdAllocator};
    use crate::object::builder::{CpuInstr, Nullary};

    use std::fmt::Debug;
    use std::iter;

    type Expr<S> = crate::model::Expr<Atom<LocationCounter, usize>, S>;

    impl<S: Session> IntoSemanticActions<S> for () {
        type SemanticActions =
            TokenStreamActionCollector<S, S::Ident, Literal<S::StringRef>, S::Span>;

        fn into_semantic_actions(self, session: S) -> Self::SemanticActions {
            TokenStreamActionCollector::new(session, panic)
        }
    }

    fn panic<I>(_: &I) -> IdentKind {
        panic!("tried annotating an identifier instead of skipping parsing")
    }

    #[test]
    fn emit_instruction_item() {
        let item = Item::CpuInstr(CpuInstr::Nullary(Nullary::Nop));
        let log =
            Fixture::<()>::default().log_session(|mut session| session.emit_item(item.clone()));
        assert_eq!(log, [BackendEvent::EmitItem(item).into()]);
    }

    #[test]
    fn define_label() {
        let label = "label";
        let log = Fixture::default().log_session(|mut session| {
            let id = session.alloc_name(());
            session.insert(label.into(), ResolvedIdent::Backend(id));
            let mut builder = session.define_symbol(id, ());
            builder.push_op(LocationCounter, ());
            builder.finish();
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
        let name: String = "my_section".into();
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
        let tokens = vec![(Ok(Token::Ident("NOP".into())), ())];
        let log = Fixture::new(|fixture| fixture.codebase.set_file(path, tokens.clone()))
            .log_session(|session| session.analyze_file(path.into(), ()).0.unwrap());
        assert_eq!(log, [ParserEvent::ParseTokenStream(tokens).into()]);
    }

    #[test]
    fn define_and_call_macro() {
        let tokens = vec![Token::Ident("NOP".into())];
        let spans: Vec<_> = iter::repeat(()).take(tokens.len()).collect();
        let log = Fixture::default().log_session(|mut session| {
            let id = session.define_macro((), (vec![], vec![]), (tokens.clone(), spans.clone()));
            session.call_macro((id, ()), (vec![], vec![]), ());
        });
        assert_eq!(
            log,
            [ParserEvent::ParseTokenStream(
                tokens.into_iter().map(|token| (Ok(token), ())).collect()
            )
            .into()]
        );
    }

    #[test]
    fn define_and_call_macro_with_param() {
        let db = Token::Ident("DB".into());
        let arg = Token::Literal(Literal::Number(0x42));
        let literal0 = Token::Literal(Literal::Number(0));
        let param = "x";
        let log = Fixture::default().log_session(|mut session| {
            let id = session.define_macro(
                (),
                (vec![param.into()], vec![()]),
                (
                    vec![db.clone(), Token::Ident(param.into()), literal0.clone()],
                    vec![(), (), ()],
                ),
            );
            session.call_macro((id, ()), (vec![vec![arg.clone()]], vec![vec![()]]), ());
        });
        assert_eq!(
            log,
            [ParserEvent::ParseTokenStream(
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
        let nop = Token::Ident("NOP".into());
        let label = "label";
        let param = "x";
        let log = Fixture::default().log_session(|mut session| {
            let id = session.define_macro(
                (),
                (vec![param.into()], vec![()]),
                (vec![Token::Label(param.into()), nop.clone()], vec![(), ()]),
            );
            session.call_macro(
                (id, ()),
                (vec![vec![Token::Ident(label.into())]], vec![vec![()]]),
                (),
            );
        });
        assert_eq!(
            log,
            [ParserEvent::ParseTokenStream(
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

    impl Default for MockSpan<&'static str> {
        fn default() -> Self {
            unreachable!()
        }
    }

    #[test]
    fn build_value_from_number() {
        Fixture::default().log_session(|session| {
            let mut builder = session.build_const();
            builder.push_op(42, ());
            let (_, value) = builder.finish();
            assert_eq!(value, 42.into())
        });
    }

    #[test]
    fn apply_operator_on_two_values() {
        Fixture::default().log_session(|session| {
            let mut builder = session.build_const();
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

    type MockParserFactory<S> = crate::analysis::syntax::parser::mock::MockParserFactory<Event<S>>;
    type MockBackend<S> = crate::object::builder::mock::MockBackend<SerialIdAllocator, Event<S>>;
    type MockDiagnosticsSystem<S> = crate::diag::MockDiagnosticsSystem<Event<S>, S>;
    type MockNameTable<S> =
        crate::analysis::resolve::MockNameTable<BasicNameTable<usize, MacroId>, Event<S>>;
    type TestSession<'a, S> = SessionComponents<
        Upstream<
            &'a mut MockCodebase<S>,
            &'a mut MockParserFactory<S>,
            String,
            String,
            <MockDiagnosticsSystem<S> as AddMacroDef<S>>::MacroDefHandle,
        >,
        MockBackend<S>,
        &'a mut MockNameTable<S>,
        &'a mut MockDiagnosticsSystem<S>,
    >;

    #[derive(Debug, PartialEq)]
    enum Event<S: Clone> {
        Parser(ParserEvent<String, Literal<String>, LexError, S>),
        Backend(BackendEvent<usize, Expr<S>>),
        NameTable(NameTableEvent<usize, MacroId>),
        Diagnostics(DiagnosticsEvent<S>),
    }

    impl<S: Clone> From<ParserEvent<String, Literal<String>, LexError, S>> for Event<S> {
        fn from(event: ParserEvent<String, Literal<String>, LexError, S>) -> Self {
            Event::Parser(event)
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
        analyzer: MockParserFactory<S>,
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
                    analyzer: MockParserFactory::new(log.clone()),
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
            f(SessionComponents::new(
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
