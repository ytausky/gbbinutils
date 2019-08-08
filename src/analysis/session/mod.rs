pub use super::backend::ValueBuilder;

use self::builder::Builder;
use self::expand::{DefineMacro, Expand, MacroId, MacroTable};

use super::backend::*;
use super::resolve::{NameTable, ResolvedIdent, StartScope};
use super::syntax::{LexError, ParseTokenStream, ParserFactory, TokenStreamActions};
use super::{IdentSource, Lex, Literal, SemanticToken, StringSource, TokenSeq};

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
    Self: IdentSource + SpanSource + StringSource,
    Self: BasicSession<<Self as IdentSource>::Ident, <Self as SpanSource>::Span>,
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

pub(super) trait BasicSession<I, S: Clone>
where
    Self: Sized,
    Self: AllocName<S>,
    Self: PartialBackend<S>,
    Self: StartSection<<Self as AllocName<S>>::Name, S>,
    Self: StartScope<I>,
    Self: NameTable<I, BackendEntry = <Self as AllocName<S>>::Name>,
    Self: Diagnostics<S>,
{
    type FnBuilder: ValueBuilder<Self::Name, S>
        + AllocName<S, Name = Self::Name>
        + NameTable<I, BackendEntry = Self::Name>
        + Finish<Parent = Self, Value = ()>
        + Diagnostics<S>;
    type GeneralBuilder: ValueBuilder<Self::Name, S>
        + AllocName<S, Name = Self::Name>
        + NameTable<I, BackendEntry = Self::Name>
        + Finish<Parent = Self, Value = Self::Value>
        + Diagnostics<S>;

    fn build_value(self) -> Self::GeneralBuilder;
    fn define_symbol(self, name: Self::Name, span: S) -> Self::FnBuilder;
}

pub(super) trait IntoSemanticActions<S: Session> {
    type SemanticActions;

    fn into_semantic_actions(self, session: S) -> Self::SemanticActions;
}

pub(super) type MacroArgs<I, R, S> = expand::MacroArgs<SemanticToken<I, R>, S>;
pub(super) type Params<I, S> = (Vec<I>, Vec<S>);

pub(super) struct SessionComponents<U, B, N, D> {
    upstream: U,
    downstream: Downstream<B, N, D>,
}

pub(super) struct Upstream<C, P, I, R, H> {
    codebase: C,
    parser_factory: P,
    macros: MacroTable<I, Literal<R>, H>,
}

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
        self.downstream.backend.emit_item(item)
    }

    fn reserve(&mut self, bytes: Self::Value) {
        self.downstream.backend.reserve(bytes)
    }

    fn set_origin(&mut self, origin: Self::Value) {
        self.downstream.backend.set_origin(origin)
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
            .lex_file(path, &mut *self.downstream.diagnostics)
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
        self.upstream.macros.define_macro(
            name_span,
            params,
            body,
            &mut *self.downstream.diagnostics,
        )
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
        let expansion =
            self.upstream.macros[id].expand(span, args, &mut *self.downstream.diagnostics);
        let mut parser = self.upstream.parser_factory.mk_parser();
        let actions = actions.into_semantic_actions(self);
        parser.parse_token_stream(expansion.map(|(t, s)| (Ok(t), s)), actions)
    }
}

impl<U, B, N, D, I, S> BasicSession<I, S> for SessionComponents<U, B, N, D>
where
    B: Backend<S>,
    N: DerefMut,
    N::Target: NameTable<I, BackendEntry = B::Name> + StartScope<I>,
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

impl<U, B: AllocName<S>, N, D, S: Clone> AllocName<S> for SessionComponents<U, B, N, D> {
    type Name = B::Name;

    fn alloc_name(&mut self, span: S) -> Self::Name {
        self.downstream.backend.alloc_name(span)
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
        self.downstream.get(ident)
    }

    fn insert(&mut self, ident: I, entry: ResolvedIdent<Self::BackendEntry, Self::MacroEntry>) {
        self.downstream.insert(ident, entry)
    }
}

delegate_diagnostics! {
    {'a, U, B, N, D: DerefMut, S: Clone},
    {D::Target: Diagnostics<S>},
    SessionComponents<U, B, N, D>,
    {downstream.diagnostics},
    D::Target,
    S
}

impl<U, B, N, D, I> StartScope<I> for SessionComponents<U, B, N, D>
where
    N: DerefMut,
    N::Target: StartScope<I>,
{
    fn start_scope(&mut self, ident: &I) {
        self.downstream.names.start_scope(ident)
    }
}

impl<U, B, N, D, S> StartSection<B::Name, S> for SessionComponents<U, B, N, D>
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

    use crate::analysis::backend::{BackendEvent, PanickingIdAllocator};
    use crate::analysis::resolve::{BasicNameTable, FakeNameTable, MockNameTable, NameTableEvent};
    use crate::diag::{DiagnosticsEvent, MockDiagnostics};
    use crate::log::Log;
    use crate::model::Atom;

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
                downstream: Downstream {
                    backend: MockBackend::new(alloc, log.clone()),
                    names: Box::new(MockNameTable::new(names, log.clone())),
                    diagnostics: Box::new(MockDiagnostics::new(log.clone())),
                },
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

    pub(in crate::analysis) type MockBuilder<A, N, T, S> = RelocContext<
        (),
        Downstream<
            RelocContext<MockBackend<A, T>, Expr<<A as AllocName<S>>::Name, S>>,
            Box<MockNameTable<N, T>>,
            Box<MockDiagnostics<T, S>>,
        >,
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
                parent: (),
                builder: Downstream {
                    backend: MockBackend::new(alloc, log.clone()).build_immediate(),
                    names: Box::new(MockNameTable::new(names, log.clone())),
                    diagnostics: Box::new(MockDiagnostics::new(log)),
                },
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

    use crate::analysis::backend::{BackendEvent, SerialIdAllocator};
    use crate::analysis::resolve::{BasicNameTable, NameTableEvent};
    use crate::analysis::syntax::*;
    use crate::analysis::{Literal, MockCodebase};
    use crate::diag::span::{MergeSpans, StripSpan};
    use crate::diag::DiagnosticsEvent;
    use crate::log::*;
    use crate::model::{Atom, BinOp, Instruction, Nullary};

    use std::fmt::Debug;
    use std::iter;

    type Expr<S> = crate::model::Expr<Atom<LocationCounter, usize>, S>;

    impl<S: Session> IntoSemanticActions<S> for () {
        type SemanticActions = MockSemanticActions<S>;

        fn into_semantic_actions(self, session: S) -> Self::SemanticActions {
            MockSemanticActions(session)
        }
    }

    pub struct MockSemanticActions<S>(S);

    impl<A, S: Clone> TokenStreamActions<String, Literal<String>, S> for MockSemanticActions<A> {
        type InstrLineActions = Self;
        type TokenLineActions = Self;
        type TokenLineFinalizer = Self;

        fn will_parse_line(self) -> LineRule<Self::InstrLineActions, Self::TokenLineActions> {
            unimplemented!()
        }

        fn act_on_eos(self, _: S) -> Self {
            unimplemented!()
        }
    }

    impl<A, S: Clone> InstrLineActions<String, Literal<String>, S> for MockSemanticActions<A> {
        type LabelActions = Self;
        type InstrActions = Self;

        fn will_parse_label(self, _: (String, S)) -> Self::LabelActions {
            unimplemented!()
        }
    }

    impl<A, S: Clone> LabelActions<String, S> for MockSemanticActions<A> {
        type Next = Self;

        fn act_on_param(&mut self, _: String, _: S) {
            unimplemented!()
        }

        fn did_parse_label(self) -> Self::Next {
            unimplemented!()
        }
    }

    impl<A, S: Clone> InstrActions<String, Literal<String>, S> for MockSemanticActions<A> {
        type BuiltinInstrActions = Self;
        type MacroInstrActions = Self;
        type ErrorActions = Self;
        type LineFinalizer = Self;

        fn will_parse_instr(
            self,
            _: String,
            _: S,
        ) -> InstrRule<Self::BuiltinInstrActions, Self::MacroInstrActions, Self::ErrorActions>
        {
            unimplemented!()
        }
    }

    impl<A, S: Clone> LineFinalizer<S> for MockSemanticActions<A> {
        type Next = Self;

        fn did_parse_line(self, _: S) -> Self::Next {
            unimplemented!()
        }
    }

    impl<A, S: Clone> InstrFinalizer<S> for MockSemanticActions<A> {
        type Next = Self;

        fn did_parse_instr(self) -> Self::Next {
            unimplemented!()
        }
    }

    impl<A, T, S: Clone> EmitDiag<S, T> for MockSemanticActions<A> {
        fn emit_diag(&mut self, _: impl Into<CompactDiag<S, T>>) {
            unimplemented!()
        }
    }

    impl<A, S: Clone> StripSpan<S> for MockSemanticActions<A> {
        type Stripped = ();

        fn strip_span(&mut self, _: &S) -> Self::Stripped {
            unimplemented!()
        }
    }

    impl<A, S: Clone> MergeSpans<S> for MockSemanticActions<A> {
        fn merge_spans(&mut self, _: &S, _: &S) -> S {
            unimplemented!()
        }
    }

    impl<A, S: Clone> BuiltinInstrActions<String, Literal<String>, S> for MockSemanticActions<A> {
        type ArgActions = Self;

        fn will_parse_arg(self) -> Self::ArgActions {
            unimplemented!()
        }
    }

    impl<A, S: Clone> ArgActions<String, Literal<String>, S> for MockSemanticActions<A> {
        fn act_on_atom(&mut self, _: ExprAtom<String, Literal<String>>, _: S) {
            unimplemented!()
        }

        fn act_on_operator(&mut self, _: Operator, _: S) {
            unimplemented!()
        }
    }

    impl<A> ArgFinalizer for MockSemanticActions<A> {
        type Next = Self;

        fn did_parse_arg(self) -> Self::Next {
            unimplemented!()
        }
    }

    impl<A, S: Clone> MacroInstrActions<S> for MockSemanticActions<A> {
        type Token = Token<String, Literal<String>>;
        type MacroArgActions = Self;

        fn will_parse_macro_arg(self) -> Self::MacroArgActions {
            unimplemented!()
        }
    }

    impl<A, S: Clone> MacroArgActions<S> for MockSemanticActions<A> {
        type Token = Token<String, Literal<String>>;
        type Next = Self;

        fn act_on_token(&mut self, _: (Self::Token, S)) {
            unimplemented!()
        }

        fn did_parse_macro_arg(self) -> Self::Next {
            unimplemented!()
        }
    }

    impl<A, S: Clone> TokenLineActions<String, Literal<String>, S> for MockSemanticActions<A> {
        type ContextFinalizer = Self;

        fn act_on_token(&mut self, _: Token<String, Literal<String>>, _: S) {
            unimplemented!()
        }

        fn act_on_ident(self, _: String, _: S) -> TokenLineRule<Self, Self::ContextFinalizer> {
            unimplemented!()
        }
    }

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

    type MockParserFactory<S> = crate::analysis::syntax::MockParserFactory<Event<S>>;
    type MockBackend<S> = crate::analysis::backend::MockBackend<SerialIdAllocator, Event<S>>;
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
