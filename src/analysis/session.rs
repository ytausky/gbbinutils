pub use super::backend::ValueBuilder;

use super::backend::*;
use super::macros::{DefineMacro, Expand, MacroEntry};
use super::semantics::Analyze;
use super::{Lex, SemanticToken, StringRef};

use crate::codebase::CodebaseError;
use crate::diag::span::{Source, Span};
use crate::diag::*;
use crate::model::{BinOp, Item, ParamId};
use crate::name::{Ident, Name, NameTable, StartScope};

#[cfg(test)]
pub(crate) use self::mock::*;

pub(crate) trait Session
where
    Self: Sized,
    Self: Span + StringRef,
    Self: DelegateDiagnostics<<Self as Span>::Span>,
    Self: PartialBackend<<Self as Span>::Span>,
    Self: StartSection<Ident<<Self as StringRef>::StringRef>, <Self as Span>::Span>,
{
    type FnBuilder: ValueBuilder<Ident<Self::StringRef>, Self::Span>
        + FinishFnDef<Return = Self>
        + DelegateDiagnostics<Self::Span>;
    type GeneralBuilder: ValueBuilder<Ident<Self::StringRef>, Self::Span>
        + Finish<Self::Span, Parent = Self, Value = Self::Value>
        + DelegateDiagnostics<Self::Span>;

    fn analyze_file(self, path: Self::StringRef) -> (Result<(), CodebaseError>, Self);
    fn build_value(self) -> Self::GeneralBuilder;
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
    fn define_symbol(&mut self, symbol: (Ident<Self::StringRef>, Self::Span), value: Self::Value);
}

pub trait Finish<S: Clone> {
    type Parent;
    type Value: Source<Span = S>;

    fn finish(self) -> (Self::Parent, Self::Value);
}

pub trait FinishFnDef {
    type Return;

    fn finish_fn_def(self) -> Self::Return;
}

pub(super) type MacroArgs<I, S> = Vec<Vec<(SemanticToken<I>, S)>>;
pub(super) type Params<R, S> = (Vec<Ident<R>>, Vec<S>);

pub(crate) struct CompositeSession<'a, 'b, C, A, B, N, D> {
    codebase: &'b mut C,
    analyzer: &'a mut A,
    backend: B,
    names: &'b mut N,
    diagnostics: &'b mut D,
}

impl<'a, 'b, C, A, B, N, D> CompositeSession<'a, 'b, C, A, B, N, D> {
    pub fn new(
        codebase: &'b mut C,
        analyzer: &'a mut A,
        backend: B,
        names: &'b mut N,
        diagnostics: &'b mut D,
    ) -> Self {
        CompositeSession {
            codebase,
            analyzer,
            backend,
            names,
            diagnostics,
        }
    }
}

impl<'a, 'b, C, A, B, N, D> CompositeSession<'a, 'b, C, A, B, N, D>
where
    C: Lex<D>,
    B: AllocName<D::Span>,
    N: NameTable<Ident<C::StringRef>, BackendEntry = B::Name>,
    D: Diagnostics,
{
    fn look_up_symbol(&mut self, ident: Ident<C::StringRef>, span: &D::Span) -> B::Name {
        match self.names.get(&ident) {
            Some(Name::Backend(id)) => id.clone(),
            Some(Name::Macro(_)) => unimplemented!(),
            None => {
                let id = self.backend.alloc_name(span.clone());
                self.names.insert(ident, Name::Backend(id.clone()));
                id
            }
        }
    }
}

pub struct PartialSession<'b, C: 'b, B, N: 'b, D: 'b> {
    pub codebase: &'b mut C,
    pub backend: B,
    pub names: &'b mut N,
    pub diagnostics: &'b mut D,
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

impl<'a, 'b, C, A, B, N, D> From<CompositeSession<'a, 'b, C, A, B, N, D>>
    for PartialSession<'b, C, B, N, D>
{
    fn from(session: CompositeSession<'a, 'b, C, A, B, N, D>) -> Self {
        partial!(session)
    }
}

impl<'a, 'b, F, A, B, N, D> Span for CompositeSession<'a, 'b, F, A, B, N, D>
where
    D: Span,
{
    type Span = D::Span;
}

impl<'a, 'b, C, A, B, N, D> StringRef for CompositeSession<'a, 'b, C, A, B, N, D>
where
    C: Lex<D>,
    D: Diagnostics,
{
    type StringRef = C::StringRef;
}

impl<'a, 'b, C, A, B, N, D> PartialBackend<D::Span> for CompositeSession<'a, 'b, C, A, B, N, D>
where
    C: Lex<D>,
    B: Backend<D::Span>,
    N: NameTable<Ident<C::StringRef>, MacroEntry = MacroEntry<C::StringRef, D>>,
    D: Diagnostics,
    B::Value: Default + ValueBuilder<B::Name, D::Span>,
{
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

pub(crate) struct RelocContext<P, B> {
    parent: P,
    builder: B,
}

macro_rules! impl_push_op_for_reloc_context {
    ($t:ty) => {
        impl<P, B, S> PushOp<$t, S> for RelocContext<P, B>
        where
            B: PushOp<$t, S>,
            S: Clone,
        {
            fn push_op(&mut self, op: $t, span: S) {
                self.builder.push_op(op, span)
            }
        }
    };
}

impl_push_op_for_reloc_context! {LocationCounter}
impl_push_op_for_reloc_context! {i32}
impl_push_op_for_reloc_context! {BinOp}
impl_push_op_for_reloc_context! {ParamId}

impl<'a, 'b, C, A, B, N, D> PushOp<Ident<C::StringRef>, D::Span>
    for RelocContext<CompositeSession<'a, 'b, C, A, B, N, D>, B::Value>
where
    C: Lex<D>,
    B: Backend<D::Span>,
    N: NameTable<Ident<C::StringRef>, BackendEntry = B::Name>,
    D: Diagnostics,
    B::Value: Default + ValueBuilder<B::Name, D::Span>,
{
    fn push_op(&mut self, ident: Ident<C::StringRef>, span: D::Span) {
        let symbol_id = self.parent.look_up_symbol(ident, &span);
        self.builder.push_op(symbol_id, span)
    }
}

impl<P, B, S> Finish<S> for RelocContext<P, B>
where
    B: Source<Span = S>,
    S: Clone,
{
    type Parent = P;
    type Value = B;

    fn finish(self) -> (Self::Parent, Self::Value) {
        (self.parent, self.builder)
    }
}

impl<'a, 'b, C, A, B, N, D> FinishFnDef
    for RelocContext<CompositeSession<'a, 'b, C, A, B, N, D>, B::Value>
where
    C: Lex<D>,
    B: Backend<D::Span>,
    D: Diagnostics,
    B::Value: Default + ValueBuilder<B::Name, D::Span>,
{
    type Return = CompositeSession<'a, 'b, C, A, B, N, D>;

    fn finish_fn_def(self) -> Self::Return {
        unimplemented!()
    }
}

impl<'a, 'b, C, A, B, N, D> DelegateDiagnostics<D::Span>
    for RelocContext<CompositeSession<'a, 'b, C, A, B, N, D>, B::Value>
where
    C: Lex<D>,
    B: Backend<D::Span>,
    D: Diagnostics,
    B::Value: Default + ValueBuilder<B::Name, D::Span>,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.diagnostics
    }
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
    D: Diagnostics,
    B::Value: Default + ValueBuilder<B::Name, D::Span>,
{
    type FnBuilder = RelocContext<Self, B::Value>;
    type GeneralBuilder = RelocContext<Self, B::Value>;

    fn analyze_file(mut self, path: Self::StringRef) -> (Result<(), CodebaseError>, Self) {
        let tokens = match self.codebase.lex_file(path, self.diagnostics) {
            Ok(tokens) => tokens,
            Err(error) => return (Err(error), self),
        };
        let PartialSession { backend, .. } =
            self.analyzer.analyze_token_seq(tokens, partial!(self));
        self.backend = backend;
        (Ok(()), self)
    }

    fn build_value(self) -> Self::GeneralBuilder {
        RelocContext {
            parent: self,
            builder: Default::default(),
        }
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

    fn call_macro(
        mut self,
        name: (Ident<Self::StringRef>, Self::Span),
        args: MacroArgs<Self::StringRef, Self::Span>,
    ) -> Self {
        let expansion = match self.names.get(&name.0) {
            Some(Name::Macro(entry)) => Some(entry.expand(name.1, args, self.diagnostics)),
            Some(_) => unimplemented!(),
            None => {
                let stripped = self.diagnostics.strip_span(&name.1);
                self.diagnostics
                    .emit_diagnostic(Message::UndefinedMacro { name: stripped }.at(name.1));
                None
            }
        };
        if let Some(expansion) = expansion {
            let PartialSession { backend, .. } = self
                .analyzer
                .analyze_token_seq(expansion.map(|(t, s)| (Ok(t), s)), partial!(self));
            self.backend = backend
        }
        self
    }

    fn define_symbol(&mut self, symbol: (Ident<Self::StringRef>, Self::Span), value: Self::Value) {
        self.names.start_scope(&symbol.0);
        let symbol_id = self.look_up_symbol(symbol.0, &symbol.1);
        self.backend.define_fn((symbol_id, symbol.1), value)
    }
}

impl<'a, 'b, F, A, B, N, D, S> DelegateDiagnostics<S> for CompositeSession<'a, 'b, F, A, B, N, D>
where
    D: DownstreamDiagnostics<S>,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.diagnostics
    }
}

impl<'a, 'b, C, A, B, N, D> StartSection<Ident<C::StringRef>, D::Span>
    for CompositeSession<'a, 'b, C, A, B, N, D>
where
    C: Lex<D>,
    B: Backend<D::Span>,
    N: NameTable<Ident<C::StringRef>, BackendEntry = B::Name>,
    D: Diagnostics,
    B::Value: Default + ValueBuilder<B::Name, D::Span>,
{
    fn start_section(&mut self, (ident, span): (Ident<C::StringRef>, D::Span)) {
        let name = self.look_up_symbol(ident, &span);
        self.backend.start_section((name, span))
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::analysis::backend::BackendEvent;
    use crate::diag::{DiagnosticsEvent, MockDiagnostics, MockSpan};
    use crate::model::{Atom, Expr};

    use std::cell::RefCell;

    #[derive(Debug, PartialEq)]
    pub(crate) enum SessionEvent<S> {
        AnalyzeFile(String),
        DefineMacro(
            Ident<String>,
            Vec<Ident<String>>,
            Vec<SemanticToken<String>>,
        ),
        InvokeMacro(Ident<String>, Vec<Vec<SemanticToken<String>>>),
        DefineSymbol((Ident<String>, S), Expr<Atom<Ident<String>>, S>),
    }

    pub(crate) struct MockSession<'a, T, S> {
        log: &'a RefCell<Vec<T>>,
        error: Option<CodebaseError>,
        diagnostics: MockDiagnostics<'a, T, S>,
        name: Option<(Ident<String>, S)>,
    }

    impl<'a, T, S> MockSession<'a, T, S> {
        pub fn new(log: &'a RefCell<Vec<T>>) -> Self {
            Self {
                log,
                error: None,
                diagnostics: MockDiagnostics::new(log),
                name: None,
            }
        }

        pub fn fail(&mut self, error: CodebaseError) {
            self.error = Some(error)
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

    type FnBuilder<S> = RelocContext<S, Expr<Atom<Ident<String>>, <S as Span>::Span>>;

    impl<'a, T, S> Session for MockSession<'a, T, S>
    where
        T: From<SessionEvent<S>>,
        T: From<BackendEvent<Expr<Atom<Ident<String>>, S>>>,
        T: From<DiagnosticsEvent<S>>,
        S: Clone + MockSpan,
    {
        type FnBuilder = FnBuilder<Self>;
        type GeneralBuilder = RelocContext<Self, Expr<Atom<Ident<String>>, S>>;

        fn analyze_file(mut self, path: String) -> (Result<(), CodebaseError>, Self) {
            self.log
                .borrow_mut()
                .push(SessionEvent::AnalyzeFile(path).into());
            (self.error.take().map_or(Ok(()), Err), self)
        }

        fn build_value(self) -> Self::GeneralBuilder {
            RelocContext {
                parent: self,
                builder: Default::default(),
            }
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

        fn call_macro(
            self,
            name: (Ident<Self::StringRef>, Self::Span),
            args: MacroArgs<Self::StringRef, Self::Span>,
        ) -> Self {
            self.log.borrow_mut().push(
                SessionEvent::InvokeMacro(
                    name.0,
                    args.into_iter()
                        .map(|arg| arg.into_iter().map(|(token, _)| token).collect())
                        .collect(),
                )
                .into(),
            );
            self
        }

        fn define_symbol(
            &mut self,
            symbol: (Ident<Self::StringRef>, Self::Span),
            value: Self::Value,
        ) {
            self.log
                .borrow_mut()
                .push(SessionEvent::DefineSymbol(symbol, value).into())
        }
    }

    impl<'a, T, S> PushOp<Ident<String>, S>
        for RelocContext<MockSession<'a, T, S>, Expr<Atom<Ident<String>>, S>>
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

    impl<'a, T, S> PushOp<Ident<String>, S>
        for RelocContext<MockDiagnostics<'a, T, S>, Expr<Atom<Ident<String>>, S>>
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

    impl<'a, T, S> DelegateDiagnostics<S>
        for RelocContext<MockSession<'a, T, S>, Expr<Atom<Ident<String>>, S>>
    where
        T: From<DiagnosticsEvent<S>>,
        S: Clone + MockSpan,
    {
        type Delegate = MockDiagnostics<'a, T, S>;

        fn diagnostics(&mut self) -> &mut Self::Delegate {
            self.parent.diagnostics()
        }
    }

    impl<'a, T, S> FinishFnDef for RelocContext<MockSession<'a, T, S>, Expr<Atom<Ident<String>>, S>>
    where
        T: From<SessionEvent<S>>,
        S: Clone,
    {
        type Return = MockSession<'a, T, S>;

        fn finish_fn_def(self) -> Self::Return {
            let mut session = self.parent;
            session.log.borrow_mut().push(
                SessionEvent::DefineSymbol(session.name.take().unwrap(), self.builder).into(),
            );
            session
        }
    }

    impl<'a, T, S> PartialBackend<S> for MockSession<'a, T, S>
    where
        T: From<BackendEvent<Expr<Atom<Ident<String>>, S>>>,
        S: Clone + MockSpan,
    {
        type Value = Expr<Atom<Ident<String>>, S>;

        fn emit_item(&mut self, item: Item<Self::Value>) {
            self.log
                .borrow_mut()
                .push(BackendEvent::EmitItem(item).into())
        }

        fn reserve(&mut self, bytes: Self::Value) {
            self.log
                .borrow_mut()
                .push(BackendEvent::Reserve(bytes).into())
        }

        fn set_origin(&mut self, origin: Self::Value) {
            self.log
                .borrow_mut()
                .push(BackendEvent::SetOrigin(origin).into())
        }
    }

    impl<'a, T, S> StartSection<Ident<String>, S> for MockSession<'a, T, S>
    where
        T: From<BackendEvent<Expr<Atom<Ident<String>>, S>>>,
        S: Clone + MockSpan,
    {
        fn start_section(&mut self, name: (Ident<String>, S)) {
            self.log
                .borrow_mut()
                .push(BackendEvent::StartSection((0, name.1)).into())
        }
    }

    pub(crate) type MockBuilder<'a, T, S> =
        RelocContext<MockDiagnostics<'a, T, S>, Expr<Atom<Ident<String>>, S>>;

    impl<'a, T, S> MockBuilder<'a, T, S> {
        pub fn with_log(log: &'a RefCell<Vec<T>>) -> Self {
            Self {
                parent: MockDiagnostics::new(log),
                builder: Default::default(),
            }
        }
    }

    impl<'a, T, S> DelegateDiagnostics<S> for MockBuilder<'a, T, S>
    where
        T: From<DiagnosticsEvent<S>>,
        S: Clone + MockSpan,
    {
        type Delegate = MockDiagnostics<'a, T, S>;

        fn diagnostics(&mut self) -> &mut Self::Delegate {
            &mut self.parent
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::backend::BackendEvent;
    use crate::analysis::semantics::AnalyzerEvent;
    use crate::analysis::{Literal, MockCodebase};
    use crate::diag::{DiagnosticsEvent, MockSpan};
    use crate::model::{Atom, Expr, Instruction, Nullary, Width};
    use crate::name::{BasicNameTable, NameTableEvent};
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
        session.define_symbol((label.into(), ()), Atom::LocationCounter.into());
        assert_eq!(
            log.into_inner(),
            [
                NameTableEvent::StartScope(label.into()).into(),
                BackendEvent::DefineSymbol((0, ()), Atom::LocationCounter.into()).into()
            ]
        );
    }

    #[test]
    fn start_section() {
        let name: Ident<_> = "my_section".into();
        let log = RefCell::new(Vec::new());
        let mut fixture = Fixture::new(&log);
        let mut session = fixture.session();
        session.start_section((name.clone(), ()));
        assert_eq!(
            log.into_inner(),
            [BackendEvent::StartSection((0, ())).into()]
        )
    }

    #[test]
    fn look_up_section_name_after_definition() {
        let ident: Ident<_> = "my_section".into();
        let log = RefCell::new(Vec::new());
        let mut fixture = Fixture::new(&log);
        let mut session = fixture.session();
        session.start_section((ident.clone(), ()));
        let mut builder = session.build_value();
        builder.push_op(ident, ());
        let (s, value) = Finish::finish(builder);
        let item = Item::Data(value, Width::Word);
        session = s;
        session.emit_item(item);
        assert_eq!(
            log.into_inner(),
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
        let log = RefCell::new(Vec::new());
        let mut fixture = Fixture::new(&log);
        fixture.codebase.set_file(path, tokens.clone());
        let session = fixture.session();
        session.analyze_file(path.into()).0.unwrap();
        assert_eq!(
            log.into_inner(),
            [AnalyzerEvent::AnalyzeTokenSeq(tokens).into()]
        );
    }

    #[test]
    fn define_and_call_macro() {
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
        session.call_macro((name.into(), ()), vec![]);
        assert_eq!(
            log.into_inner(),
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
        session.call_macro((name.into(), ()), vec![vec![(arg.clone(), ())]]);
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
    fn define_and_call_macro_with_label() {
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
        session.call_macro(
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
    fn reserve_bytes() {
        let bytes = 10;
        let log = RefCell::new(Vec::new());
        let mut fixture = Fixture::new(&log);
        let mut session = fixture.session();
        session.reserve(bytes.into());
        assert_eq!(
            log.into_inner(),
            [BackendEvent::Reserve(bytes.into()).into()]
        )
    }

    #[test]
    fn diagnose_undefined_macro() {
        let name = "my_macro";
        let log = RefCell::new(Vec::new());
        let mut fixture = Fixture::new(&log);
        let session = fixture.session();
        session.call_macro((name.into(), name), vec![]);
        assert_eq!(
            log.into_inner(),
            [
                DiagnosticsEvent::EmitDiagnostic(Message::UndefinedMacro { name }.at(name).into())
                    .into()
            ]
        );
    }

    #[test]
    fn build_value_from_number() {
        let log = RefCell::new(Vec::new());
        let mut fixture = Fixture::new(&log);
        let session = fixture.session();
        let mut builder = session.build_value();
        builder.push_op(42, ());
        let (_, value) = builder.finish();
        assert_eq!(value, 42.into())
    }

    #[test]
    fn apply_operator_on_two_values() {
        let log = RefCell::new(Vec::new());
        let mut fixture = Fixture::new(&log);
        let session = fixture.session();
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
    }

    type MockAnalyzer<'a, S> = crate::analysis::semantics::MockAnalyzer<'a, Event<S>>;
    type MockBackend<'a, S> = crate::analysis::backend::MockBackend<'a, Event<S>>;
    type MockDiagnostics<'a, S> = crate::diag::MockDiagnostics<'a, Event<S>, S>;
    type MockNameTable<'a, S> = crate::name::MockNameTable<
        'a,
        BasicNameTable<usize, MacroEntry<String, MockDiagnostics<'a, S>>>,
        Event<S>,
    >;
    type TestSession<'a, 'b, S> = CompositeSession<
        'b,
        'b,
        MockCodebase<S>,
        MockAnalyzer<'a, S>,
        MockBackend<'a, S>,
        MockNameTable<'a, S>,
        MockDiagnostics<'a, S>,
    >;

    #[derive(Debug, PartialEq)]
    enum Event<S: Clone> {
        Frontend(AnalyzerEvent<S>),
        Backend(BackendEvent<Expr<Atom<usize>, S>>),
        NameTable(NameTableEvent),
        Diagnostics(DiagnosticsEvent<S>),
    }

    impl<S: Clone> From<AnalyzerEvent<S>> for Event<S> {
        fn from(event: AnalyzerEvent<S>) -> Self {
            Event::Frontend(event)
        }
    }

    impl<S: Clone> From<BackendEvent<Expr<Atom<usize>, S>>> for Event<S> {
        fn from(event: BackendEvent<Expr<Atom<usize>, S>>) -> Self {
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

    struct Fixture<'a, S: Clone + MockSpan> {
        codebase: MockCodebase<S>,
        analyzer: MockAnalyzer<'a, S>,
        backend: Option<MockBackend<'a, S>>,
        names: MockNameTable<'a, S>,
        diagnostics: MockDiagnostics<'a, S>,
    }

    impl<'a, S: Clone + MockSpan> Fixture<'a, S> {
        fn new(log: &'a RefCell<Vec<Event<S>>>) -> Self {
            Self {
                codebase: MockCodebase::new(),
                analyzer: MockAnalyzer::new(log),
                backend: Some(MockBackend::new(log)),
                names: MockNameTable::new(BasicNameTable::new(), log),
                diagnostics: MockDiagnostics::new(log),
            }
        }

        fn session<'b>(&'b mut self) -> TestSession<'a, 'b, S> {
            CompositeSession::new(
                &mut self.codebase,
                &mut self.analyzer,
                self.backend.take().unwrap(),
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
