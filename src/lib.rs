use std::{borrow::{Borrow, BorrowMut}, fmt::Debug, marker::PhantomData};

mod backend;
mod codebase;
mod diagnostics;
mod frontend;

pub fn analyze_file(name: &str) {
    let codebase = codebase::FileCodebase::new(codebase::StdFileSystem::new());
    let diagnostics = diagnostics::TerminalDiagnostics::new(&codebase.cache);
    frontend::analyze_file(
        name.to_string(),
        &codebase,
        diagnostics::SimpleTokenTracker {},
        OutputDumper::new(),
        &diagnostics,
    );
}

pub fn assemble_rom(name: &str) {
    let codebase = codebase::FileCodebase::new(codebase::StdFileSystem::new());
    let diagnostics = diagnostics::TerminalDiagnostics::new(&codebase.cache);
    frontend::analyze_file(
        name.to_string(),
        &codebase,
        diagnostics::SimpleTokenTracker {},
        backend::ObjectBuilder::new(&diagnostics),
        &diagnostics,
    );
}

pub trait AssemblySession {
    type TokenRef: Debug + PartialEq;
    fn analyze_chunk(&mut self, chunk_id: ChunkId<Self::TokenRef>);
    fn emit_diagnostic(&mut self, diagnostic: diagnostics::Diagnostic<Self::TokenRef>);
    fn emit_item(&mut self, item: backend::Item<Self::TokenRef>);
    fn define_label(&mut self, label: (String, Self::TokenRef));
    fn define_macro(
        &mut self,
        name: (impl Into<String>, Self::TokenRef),
        tokens: Vec<(frontend::Token, Self::TokenRef)>,
    );
}

#[derive(Debug, PartialEq)]
pub enum ChunkId<T> {
    File((String, Option<T>)),
    Macro {
        name: (String, T),
        args: Vec<Vec<frontend::Token>>,
    },
}

struct Session<R, F, B, D, BMF, BMB, BD>
where
    F: frontend::Frontend,
    B: backend::Backend<R>,
    D: diagnostics::DiagnosticsListener<R>,
    BMF: BorrowMut<F>,
    BMB: BorrowMut<B>,
    BD: Borrow<D>,
{
    frontend: BMF,
    backend: BMB,
    diagnostics: BD,
    phantom: PhantomData<(R, F, B, D)>,
}

impl<R, F, B, D, BMF, BMB, BD> Session<R, F, B, D, BMF, BMB, BD>
where
    R: Debug + PartialEq,
    F: frontend::Frontend<TokenRef = R>,
    B: backend::Backend<R>,
    D: diagnostics::DiagnosticsListener<R>,
    BMF: BorrowMut<F>,
    BMB: BorrowMut<B>,
    BD: Borrow<D>,
{
    fn new(frontend: BMF, backend: BMB, diagnostics: BD) -> Session<R, F, B, D, BMF, BMB, BD> {
        Session {
            frontend,
            backend,
            diagnostics,
            phantom: PhantomData,
        }
    }
}

impl<R, F, B, D, BMF, BMB, BD> AssemblySession for Session<R, F, B, D, BMF, BMB, BD>
where
    R: Debug + PartialEq,
    F: frontend::Frontend<TokenRef = R>,
    B: backend::Backend<R>,
    D: diagnostics::DiagnosticsListener<R>,
    BMF: BorrowMut<F>,
    BMB: BorrowMut<B>,
    BD: Borrow<D>,
{
    type TokenRef = R;

    fn analyze_chunk(&mut self, chunk_id: ChunkId<Self::TokenRef>) {
        self.frontend
            .borrow_mut()
            .analyze_chunk(chunk_id, self.backend.borrow_mut())
    }

    fn emit_diagnostic(&mut self, diagnostic: diagnostics::Diagnostic<Self::TokenRef>) {
        self.diagnostics.borrow().emit_diagnostic(diagnostic)
    }

    fn emit_item(&mut self, item: backend::Item<Self::TokenRef>) {
        self.backend.borrow_mut().emit_item(item)
    }

    fn define_label(&mut self, label: (String, Self::TokenRef)) {
        self.backend.borrow_mut().add_label(label)
    }

    fn define_macro(
        &mut self,
        name: (impl Into<String>, Self::TokenRef),
        tokens: Vec<(frontend::Token, Self::TokenRef)>,
    ) {
        self.frontend.borrow_mut().define_macro(name, tokens)
    }
}

struct OutputDumper;

impl OutputDumper {
    pub fn new() -> OutputDumper {
        OutputDumper {}
    }
}

impl<R: Debug> backend::Backend<R> for OutputDumper {
    type Section = Self;
    fn mk_section(&mut self) -> Self::Section {
        OutputDumper {}
    }

    fn add_label(&mut self, (label, _): (impl Into<String>, R)) {
        println!("Define symbol: {}", label.into())
    }

    fn emit_item(&mut self, item: backend::Item<R>) {
        println!("Emit {:?}", item)
    }
}

impl<R: Debug> backend::Section<R> for OutputDumper {
    fn add_label(&mut self, (label, _): (impl Into<String>, R)) {
        println!("Define symbol: {}", label.into())
    }

    fn emit_item(&mut self, item: backend::Item<R>) {
        println!("Emit {:?}", item)
    }
}
