use crate::backend::{self, Backend, LocationCounter, ToValue, ValueBuilder};
use crate::diag::span::{MergeSpans, Source, StripSpan};
use crate::diag::*;
use crate::expr::ExprVariant;
use crate::frontend::session::Session;
use crate::frontend::syntax::{self, keyword::*, ExprAtom, Operator, UnaryOperator};
use crate::frontend::{Frontend, Ident, Literal, SemanticToken};

mod directive;
mod instruction;
mod operand;

mod expr {
    #[cfg(test)]
    use crate::expr::ExprVariant;
    use crate::expr::{BinaryOperator, Expr};
    use crate::frontend::{Ident, Literal};

    #[derive(Debug, PartialEq)]
    pub(super) enum SemanticAtom<I> {
        Ident(Ident<I>),
        Literal(Literal<I>),
    }

    impl<I> From<Literal<I>> for SemanticAtom<I> {
        fn from(literal: Literal<I>) -> Self {
            SemanticAtom::Literal(literal)
        }
    }

    #[derive(Debug, PartialEq)]
    pub enum SemanticUnary {
        Parentheses,
    }

    pub(super) type SemanticExpr<I, S> = Expr<SemanticAtom<I>, SemanticUnary, BinaryOperator, S>;

    #[cfg(test)]
    pub(super) type SemanticExprVariant<I, S> =
        ExprVariant<SemanticAtom<I>, SemanticUnary, BinaryOperator, S>;
}

use self::expr::*;

pub(crate) struct SemanticActions<'a, F: Frontend<D>, B: ?Sized, N, D: Diagnostics> {
    session: Session<'a, F, B, N, D>,
    label: Option<(Ident<F::StringRef>, D::Span)>,
}

impl<'a, F, B, N, D> SemanticActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    D: Diagnostics,
{
    pub fn new(session: Session<'a, F, B, N, D>) -> SemanticActions<'a, F, B, N, D> {
        SemanticActions {
            session,
            label: None,
        }
    }

    fn define_label_if_present(&mut self) {
        if let Some((label, span)) = self.label.take() {
            let value = {
                let mut builder = self.session.backend.build_value(self.session.names);
                ToValue::<LocationCounter, D::Span>::to_value(
                    &mut builder,
                    (LocationCounter, span.clone()),
                )
            };
            self.session
                .backend
                .define_symbol((label, span), value, self.session.names)
        }
    }
}

impl<'a, F, B, N, D> DelegateDiagnostics<D::Span> for SemanticActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.session.diagnostics
    }
}

impl<'a, F, B, N, D>
    syntax::FileContext<Ident<F::StringRef>, Literal<F::StringRef>, Command, D::Span>
    for SemanticActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    D: Diagnostics,
{
    type StmtContext = Self;

    fn enter_stmt(mut self, label: Option<(Ident<F::StringRef>, D::Span)>) -> Self::StmtContext {
        self.label = label;
        self
    }
}

impl<'a, F, B, N, D>
    syntax::StmtContext<Ident<F::StringRef>, Literal<F::StringRef>, Command, D::Span>
    for SemanticActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    D: Diagnostics,
{
    type CommandContext = CommandActions<'a, F, B, N, D>;
    type MacroParamsContext = MacroDefActions<'a, F, B, N, D>;
    type MacroInvocationContext = MacroInvocationActions<'a, F, B, N, D>;
    type Parent = Self;

    fn enter_command(self, name: (Command, D::Span)) -> Self::CommandContext {
        CommandActions::new(name, self)
    }

    fn enter_macro_def(mut self, keyword: D::Span) -> Self::MacroParamsContext {
        if self.label.is_none() {
            self.diagnostics()
                .emit_diagnostic(CompactDiagnostic::new(Message::MacroRequiresName, keyword))
        }
        MacroDefActions::new(self.label.take(), self)
    }

    fn enter_macro_invocation(
        mut self,
        name: (Ident<F::StringRef>, D::Span),
    ) -> Self::MacroInvocationContext {
        self.define_label_if_present();
        MacroInvocationActions::new(name, self)
    }

    fn exit(mut self) -> Self::Parent {
        self.define_label_if_present();
        self
    }
}

pub(crate) struct CommandActions<'a, F: Frontend<D>, B: ?Sized, N, D: Diagnostics> {
    name: (Command, D::Span),
    args: CommandArgs<F::StringRef, D::Span>,
    parent: SemanticActions<'a, F, B, N, D>,
    has_errors: bool,
}

type CommandArgs<I, S> = Vec<SemanticExpr<I, S>>;

impl<'a, F: Frontend<D>, B: ?Sized, N, D: Diagnostics> CommandActions<'a, F, B, N, D> {
    fn new(
        name: (Command, D::Span),
        parent: SemanticActions<'a, F, B, N, D>,
    ) -> CommandActions<'a, F, B, N, D> {
        CommandActions {
            name,
            args: Vec::new(),
            parent,
            has_errors: false,
        }
    }
}

impl<'a, F, B, N, D> MergeSpans<D::Span> for CommandActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    fn merge_spans(&mut self, left: &D::Span, right: &D::Span) -> D::Span {
        self.parent.diagnostics().merge_spans(left, right)
    }
}

impl<'a, F, B, N, D> StripSpan<D::Span> for CommandActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type Stripped = D::Stripped;

    fn strip_span(&mut self, span: &D::Span) -> Self::Stripped {
        self.parent.diagnostics().strip_span(span)
    }
}

impl<'a, F, B, N, D> EmitDiagnostic<D::Span, D::Stripped> for CommandActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<D::Span, D::Stripped>) {
        self.has_errors = true;
        self.parent.diagnostics().emit_diagnostic(diagnostic)
    }
}

impl<'a, F, B, N, D> DelegateDiagnostics<D::Span> for CommandActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type Delegate = Self;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self
    }
}

impl<'a, F, B, N, D> syntax::CommandContext<D::Span> for CommandActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    D: Diagnostics,
{
    type Ident = Ident<F::StringRef>;
    type Command = Command;
    type Literal = Literal<F::StringRef>;
    type ArgContext = ExprContext<'a, F, B, N, D>;
    type Parent = SemanticActions<'a, F, B, N, D>;

    fn add_argument(self) -> Self::ArgContext {
        ExprContext {
            stack: Vec::new(),
            parent: self,
        }
    }

    fn exit(mut self) -> Self::Parent {
        if !self.has_errors {
            match self.name {
                (Command::Directive(directive), span) => {
                    if !directive.requires_symbol() {
                        self.parent.define_label_if_present()
                    }
                    directive::analyze_directive((directive, span), self.args, &mut self.parent)
                }
                (Command::Mnemonic(mnemonic), range) => {
                    self.parent.define_label_if_present();
                    analyze_mnemonic((mnemonic, range), self.args, &mut self.parent)
                }
            };
        }
        self.parent
    }
}

impl Directive {
    fn requires_symbol(self) -> bool {
        match self {
            Directive::Equ => true,
            _ => false,
        }
    }
}

pub(crate) struct ExprContext<'a, F: Frontend<D>, B: ?Sized, N, D: Diagnostics> {
    stack: Vec<SemanticExpr<F::StringRef, D::Span>>,
    parent: CommandActions<'a, F, B, N, D>,
}

impl<'a, F, B, N, D> DelegateDiagnostics<D::Span> for ExprContext<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type Delegate = CommandActions<'a, F, B, N, D>;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        &mut self.parent
    }
}

impl<'a, F, B, N, D> syntax::ExprContext<D::Span> for ExprContext<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type Ident = Ident<F::StringRef>;
    type Literal = Literal<F::StringRef>;
    type Parent = CommandActions<'a, F, B, N, D>;

    fn push_atom(&mut self, atom: (ExprAtom<Self::Ident, Self::Literal>, D::Span)) {
        self.stack.push(SemanticExpr {
            variant: ExprVariant::Atom(match atom.0 {
                ExprAtom::Ident(ident) => SemanticAtom::Ident(ident),
                ExprAtom::Literal(literal) => SemanticAtom::Literal(literal),
            }),
            span: atom.1,
        })
    }

    fn apply_operator(&mut self, operator: (Operator, D::Span)) {
        match operator.0 {
            Operator::Unary(UnaryOperator::Parentheses) => {
                let inner = self.stack.pop().unwrap_or_else(|| unreachable!());
                self.stack.push(SemanticExpr {
                    variant: ExprVariant::Unary(SemanticUnary::Parentheses, Box::new(inner)),
                    span: operator.1,
                })
            }
            Operator::Binary(binary) => {
                let rhs = self.stack.pop().unwrap_or_else(|| unreachable!());
                let lhs = self.stack.pop().unwrap_or_else(|| unreachable!());
                self.stack.push(SemanticExpr {
                    variant: ExprVariant::Binary(binary, Box::new(lhs), Box::new(rhs)),
                    span: operator.1,
                })
            }
        }
    }

    fn exit(mut self) -> Self::Parent {
        if !self.parent.has_errors {
            assert_eq!(self.stack.len(), 1);
            self.parent.args.push(self.stack.pop().unwrap());
        }
        self.parent
    }
}

fn analyze_mnemonic<'a, F, B, N, D>(
    name: (Mnemonic, D::Span),
    args: CommandArgs<F::StringRef, D::Span>,
    actions: &mut SemanticActions<'a, F, B, N, D>,
) where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    D: Diagnostics,
{
    let result = instruction::analyze_instruction(
        name,
        args.into_iter(),
        ValueContext::new(
            &mut actions.session.backend.build_value(actions.session.names),
            actions.session.diagnostics,
        ),
    );
    if let Ok(instruction) = result {
        actions
            .session
            .backend
            .emit_item(backend::Item::Instruction(instruction))
    }
}

pub(crate) struct MacroDefActions<'a, F: Frontend<D>, B: ?Sized, N, D: Diagnostics> {
    name: Option<(Ident<F::StringRef>, D::Span)>,
    params: (Vec<Ident<F::StringRef>>, Vec<D::Span>),
    tokens: (Vec<SemanticToken<F::StringRef>>, Vec<D::Span>),
    parent: SemanticActions<'a, F, B, N, D>,
}

impl<'a, F: Frontend<D>, B: ?Sized, N, D: Diagnostics> MacroDefActions<'a, F, B, N, D> {
    fn new(
        name: Option<(Ident<F::StringRef>, D::Span)>,
        parent: SemanticActions<'a, F, B, N, D>,
    ) -> MacroDefActions<'a, F, B, N, D> {
        MacroDefActions {
            name,
            params: (Vec::new(), Vec::new()),
            tokens: (Vec::new(), Vec::new()),
            parent,
        }
    }
}

impl<'a, F, B, N, D> DelegateDiagnostics<D::Span> for MacroDefActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.diagnostics()
    }
}

impl<'a, F, B, N, D> syntax::MacroParamsContext<D::Span> for MacroDefActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type Ident = Ident<F::StringRef>;
    type Command = Command;
    type Literal = Literal<F::StringRef>;
    type MacroBodyContext = Self;
    type Parent = SemanticActions<'a, F, B, N, D>;

    fn add_parameter(&mut self, (param, span): (Self::Ident, D::Span)) {
        self.params.0.push(param);
        self.params.1.push(span)
    }

    fn exit(self) -> Self::MacroBodyContext {
        self
    }
}

impl<'a, F, B, N, D> syntax::TokenSeqContext<D::Span> for MacroDefActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type Token = SemanticToken<F::StringRef>;
    type Parent = SemanticActions<'a, F, B, N, D>;

    fn push_token(&mut self, (token, span): (Self::Token, D::Span)) {
        self.tokens.0.push(token);
        self.tokens.1.push(span)
    }

    fn exit(self) -> Self::Parent {
        if let Some(name) = self.name {
            let context = self.parent.session.diagnostics.add_macro_def(
                name.1.clone(),
                self.params.1.clone(),
                self.tokens.1.clone(),
            );
            self.parent
                .session
                .frontend
                .define_macro(name.0, self.params.0, self.tokens.0, context)
        }
        self.parent
    }
}

pub(crate) struct MacroInvocationActions<'a, F: Frontend<D>, B: ?Sized, N, D: Diagnostics> {
    name: (Ident<F::StringRef>, D::Span),
    args: Vec<super::TokenSeq<F::StringRef, D::Span>>,
    parent: SemanticActions<'a, F, B, N, D>,
}

impl<'a, F: Frontend<D>, B: ?Sized, N, D: Diagnostics> MacroInvocationActions<'a, F, B, N, D> {
    fn new(
        name: (Ident<F::StringRef>, D::Span),
        parent: SemanticActions<'a, F, B, N, D>,
    ) -> MacroInvocationActions<'a, F, B, N, D> {
        MacroInvocationActions {
            name,
            args: Vec::new(),
            parent,
        }
    }

    fn push_arg(&mut self, arg: Vec<(SemanticToken<F::StringRef>, D::Span)>) {
        self.args.push(arg)
    }
}

impl<'a, F, B, N, D> DelegateDiagnostics<D::Span> for MacroInvocationActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.diagnostics()
    }
}

impl<'a, F, B, N, D> syntax::MacroInvocationContext<D::Span>
    for MacroInvocationActions<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
    D: Diagnostics,
{
    type Token = SemanticToken<F::StringRef>;
    type Parent = SemanticActions<'a, F, B, N, D>;
    type MacroArgContext = MacroArgContext<'a, F, B, N, D>;

    fn enter_macro_arg(self) -> Self::MacroArgContext {
        MacroArgContext::new(self)
    }

    fn exit(mut self) -> Self::Parent {
        self.parent.session.invoke_macro(self.name, self.args);
        self.parent
    }
}

pub(crate) struct MacroArgContext<'a, F: Frontend<D>, B: ?Sized, N, D: Diagnostics> {
    tokens: Vec<(SemanticToken<F::StringRef>, D::Span)>,
    parent: MacroInvocationActions<'a, F, B, N, D>,
}

impl<'a, F: Frontend<D>, B: ?Sized, N, D: Diagnostics> MacroArgContext<'a, F, B, N, D> {
    fn new(parent: MacroInvocationActions<'a, F, B, N, D>) -> MacroArgContext<'a, F, B, N, D> {
        MacroArgContext {
            tokens: Vec::new(),
            parent,
        }
    }
}

impl<'a, F, B, N, D> DelegateDiagnostics<D::Span> for MacroArgContext<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.parent.diagnostics()
    }
}

impl<'a, F, B, N, D> syntax::TokenSeqContext<D::Span> for MacroArgContext<'a, F, B, N, D>
where
    F: Frontend<D>,
    B: ?Sized,
    D: Diagnostics,
{
    type Token = SemanticToken<F::StringRef>;
    type Parent = MacroInvocationActions<'a, F, B, N, D>;

    fn push_token(&mut self, token: (Self::Token, D::Span)) {
        self.tokens.push(token)
    }

    fn exit(mut self) -> Self::Parent {
        self.parent.push_arg(self.tokens);
        self.parent
    }
}

pub struct ValueContext<'a, B: 'a, D: 'a> {
    builder: &'a mut B,
    diagnostics: &'a mut D,
}

impl<'a, B: 'a, D: 'a> ValueContext<'a, B, D> {
    fn new(builder: &'a mut B, diagnostics: &'a mut D) -> Self {
        ValueContext {
            builder,
            diagnostics,
        }
    }
}

impl<'a, B, D, S> DelegateDiagnostics<S> for ValueContext<'a, B, D>
where
    B: 'a,
    D: DownstreamDiagnostics<S> + 'a,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.diagnostics
    }
}

trait AnalyzeExpr<I, S: Clone> {
    type Value: Source<Span = S>;

    fn analyze_expr(&mut self, expr: SemanticExpr<I, S>) -> Result<Self::Value, ()>;
}

impl<'a, I, B, D, S> AnalyzeExpr<I, S> for ValueContext<'a, B, D>
where
    B: ValueBuilder<Ident<I>, S>,
    D: DownstreamDiagnostics<S>,
    S: Clone,
{
    type Value = B::Value;

    fn analyze_expr(&mut self, expr: SemanticExpr<I, S>) -> Result<Self::Value, ()> {
        match expr.variant {
            ExprVariant::Atom(SemanticAtom::Ident(ident)) => {
                Ok(self.builder.to_value((ident, expr.span)))
            }
            ExprVariant::Atom(SemanticAtom::Literal(Literal::Number(n))) => {
                Ok(self.builder.to_value((n, expr.span)))
            }
            ExprVariant::Atom(SemanticAtom::Literal(Literal::Operand(_))) => {
                Err(CompactDiagnostic::new(
                    Message::KeywordInExpr {
                        keyword: self.diagnostics.strip_span(&expr.span),
                    },
                    expr.span,
                ))
            }
            ExprVariant::Atom(SemanticAtom::Literal(Literal::String(_))) => Err(
                CompactDiagnostic::new(Message::StringInInstruction, expr.span),
            ),
            ExprVariant::Unary(SemanticUnary::Parentheses, expr) => Ok(self.analyze_expr(*expr)?),
            ExprVariant::Binary(binary, left, right) => {
                let left = self.analyze_expr(*left)?;
                let right = self.analyze_expr(*right)?;
                Ok(self
                    .builder
                    .apply_binary_operator((binary, expr.span), left, right))
            }
        }
        .map_err(|diagnostic| {
            self.diagnostics.emit_diagnostic(diagnostic);
        })
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TokenId {
    Mnemonic,
    Operand(usize, usize),
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub struct TokenSpan {
    first: TokenId,
    last: TokenId,
}

#[cfg(test)]
pub struct DiagnosticsCollector<S>(Vec<CompactDiagnostic<S, S>>);

#[cfg(test)]
impl MergeSpans<TokenSpan> for DiagnosticsCollector<TokenSpan> {
    fn merge_spans(&mut self, left: &TokenSpan, right: &TokenSpan) -> TokenSpan {
        TokenSpan::merge(left, right)
    }
}

#[cfg(test)]
impl<S: Clone> StripSpan<S> for DiagnosticsCollector<S> {
    type Stripped = S;

    fn strip_span(&mut self, span: &S) -> Self::Stripped {
        span.clone()
    }
}

#[cfg(test)]
impl<S: Clone> EmitDiagnostic<S, S> for DiagnosticsCollector<S> {
    fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<S, S>) {
        self.0.push(diagnostic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend;
    use crate::backend::{HashMapNameTable, RelocAtom, Width};
    use crate::codebase::CodebaseError;
    use crate::diag;
    use crate::diag::{CompactDiagnostic, Message};
    use crate::expr::BinaryOperator;
    use crate::frontend::macros::{MacroDefData, MacroTableEntry};
    use crate::frontend::syntax::{
        keyword::Operand, CommandContext, ExprContext, FileContext, MacroInvocationContext,
        MacroParamsContext, StmtContext, Token, TokenSeqContext,
    };
    use crate::frontend::{Downstream, LexItem, MacroArgs};
    use std::borrow::Borrow;
    use std::cell::RefCell;
    use std::rc::Rc;

    pub(crate) struct TestFrontend<'a> {
        operations: &'a RefCell<Vec<TestOperation>>,
        error: Option<CodebaseError>,
    }

    impl<'a> TestFrontend<'a> {
        pub fn new(operations: &'a RefCell<Vec<TestOperation>>) -> Self {
            TestFrontend {
                operations,
                error: None,
            }
        }

        pub fn fail(&mut self, error: CodebaseError) {
            self.error = Some(error)
        }
    }

    impl<'a> Frontend<MockDiagnostics<'a>> for TestFrontend<'a> {
        type StringRef = String;
        type MacroDefId = usize;

        fn analyze_file<B, N>(
            &mut self,
            path: Self::StringRef,
            _downstream: Downstream<B, N, MockDiagnostics<'a>>,
        ) -> Result<(), CodebaseError>
        where
            B: Backend<Ident<String>, (), N> + ?Sized,
        {
            self.operations
                .borrow_mut()
                .push(TestOperation::AnalyzeFile(path));
            match self.error.take() {
                Some(error) => Err(error),
                None => Ok(()),
            }
        }

        fn analyze_token_seq<I, B, N>(
            &mut self,
            _tokens: I,
            _downstream: &mut Downstream<B, N, MockDiagnostics<'a>>,
        ) where
            I: IntoIterator<Item = LexItem<Self::StringRef, ()>>,
            B: Backend<Ident<Self::StringRef>, (), N> + ?Sized,
        {
            unimplemented!()
        }

        fn invoke_macro<B, N>(
            &mut self,
            name: (Ident<Self::StringRef>, ()),
            args: MacroArgs<Self::StringRef, ()>,
            _downstream: Downstream<B, N, MockDiagnostics<'a>>,
        ) where
            B: Backend<Ident<String>, (), N> + ?Sized,
        {
            self.operations
                .borrow_mut()
                .push(TestOperation::InvokeMacro(
                    name.0,
                    args.into_iter()
                        .map(|arg| arg.into_iter().map(|(token, _)| token).collect())
                        .collect(),
                ))
        }

        fn define_macro(
            &mut self,
            name: Ident<Self::StringRef>,
            params: Vec<Ident<Self::StringRef>>,
            tokens: Vec<SemanticToken<Self::StringRef>>,
            _context: usize,
        ) {
            self.operations
                .borrow_mut()
                .push(TestOperation::DefineMacro(name, params, tokens))
        }
    }

    #[derive(Debug, PartialEq)]
    pub(crate) enum TestOperation {
        AnalyzeFile(String),
        InvokeMacro(Ident<String>, Vec<Vec<SemanticToken<String>>>),
        DefineMacro(
            Ident<String>,
            Vec<Ident<String>>,
            Vec<SemanticToken<String>>,
        ),
        Backend(backend::Event<RelocExpr>),
        Diagnostics(diag::Event<()>),
    }

    type RelocExpr = backend::RelocExpr<Ident<String>, ()>;

    impl<'a> From<backend::Event<RelocExpr>> for TestOperation {
        fn from(event: backend::Event<RelocExpr>) -> Self {
            TestOperation::Backend(event)
        }
    }

    impl<'a> From<diag::Event<()>> for TestOperation {
        fn from(event: diag::Event<()>) -> Self {
            TestOperation::Diagnostics(event)
        }
    }

    #[test]
    fn emit_ld_b_deref_hl() {
        use crate::instruction::*;
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions
                .enter_stmt(None)
                .enter_command((Command::Mnemonic(Mnemonic::Ld), ()));
            let mut arg1 = command.add_argument();
            arg1.push_atom((ExprAtom::Literal(Literal::Operand(Operand::B)), ()));
            command = arg1.exit();
            let mut arg2 = command.add_argument();
            arg2.push_atom((ExprAtom::Literal(Literal::Operand(Operand::Hl)), ()));
            arg2.apply_operator((Operator::Unary(UnaryOperator::Parentheses), ()));
            arg2.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [
                backend::Event::EmitItem(backend::Item::Instruction(Instruction::Ld(Ld::Simple(
                    SimpleOperand::B,
                    SimpleOperand::DerefHl
                ))))
                .into()
            ]
        )
    }

    #[test]
    fn emit_rst_1_minus_1() {
        test_rst_1_op_1(BinaryOperator::Minus)
    }

    #[test]
    fn emit_rst_1_plus_1() {
        test_rst_1_op_1(BinaryOperator::Plus)
    }

    fn test_rst_1_op_1(op: BinaryOperator) {
        use crate::instruction::*;
        let actions = collect_semantic_actions(|actions| {
            let command = actions
                .enter_stmt(None)
                .enter_command((Command::Mnemonic(Mnemonic::Rst), ()));
            let mut expr = command.add_argument();
            expr.push_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.push_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.apply_operator((Operator::Binary(op), ()));
            expr.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [
                backend::Event::EmitItem(backend::Item::Instruction(Instruction::Rst(
                    ExprVariant::Binary(op, Box::new(1.into()), Box::new(1.into()),).into()
                )))
                .into()
            ]
        )
    }

    #[test]
    fn emit_label_word() {
        let label = "my_label";
        let actions = collect_semantic_actions(|actions| {
            let mut arg = actions
                .enter_stmt(None)
                .enter_command((Command::Directive(Directive::Dw), ()))
                .add_argument();
            arg.push_atom((ExprAtom::Ident(label.into()), ()));
            arg.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [backend::Event::EmitItem(backend::Item::Data(
                RelocAtom::Symbol(label.into()).into(),
                Width::Word
            ))
            .into()]
        );
    }

    #[test]
    fn analyze_label() {
        let label = "label";
        let actions =
            collect_semantic_actions(|actions| actions.enter_stmt(Some((label.into(), ()))).exit());
        assert_eq!(
            actions,
            [
                backend::Event::DefineSymbol((label.into(), ()), RelocAtom::LocationCounter.into())
                    .into()
            ]
        )
    }

    #[test]
    fn define_nullary_macro() {
        test_macro_definition(
            "my_macro",
            [],
            [
                Token::Command(Command::Mnemonic(Mnemonic::Xor)),
                Token::Literal(Literal::Operand(Operand::A)),
            ],
        )
    }

    #[test]
    fn define_unary_macro() {
        let param = "reg";
        test_macro_definition(
            "my_xor",
            [param],
            [
                Token::Command(Command::Mnemonic(Mnemonic::Xor)),
                Token::Ident(param.into()),
            ],
        )
    }

    #[test]
    fn define_nameless_macro() {
        let actions = collect_semantic_actions(|actions| {
            let params = actions.enter_stmt(None).enter_macro_def(());
            TokenSeqContext::exit(MacroParamsContext::exit(params))
        });
        assert_eq!(
            actions,
            [
                diag::Event::EmitDiagnostic(CompactDiagnostic::new(Message::MacroRequiresName, ()))
                    .into()
            ]
        )
    }

    fn test_macro_definition(
        name: &str,
        params: impl Borrow<[&'static str]>,
        body: impl Borrow<[SemanticToken<String>]>,
    ) {
        let actions = collect_semantic_actions(|actions| {
            let mut params_actions = actions
                .enter_stmt(Some((name.into(), ())))
                .enter_macro_def(());
            for param in params.borrow().iter().map(|&t| (t.into(), ())) {
                params_actions.add_parameter(param)
            }
            let mut token_seq_actions = MacroParamsContext::exit(params_actions);
            for token in body.borrow().iter().cloned().map(|t| (t, ())) {
                token_seq_actions.push_token(token)
            }
            TokenSeqContext::exit(token_seq_actions)
        });
        assert_eq!(
            actions,
            [TestOperation::DefineMacro(
                name.into(),
                params.borrow().iter().cloned().map(Into::into).collect(),
                body.borrow().iter().cloned().collect()
            )]
        )
    }

    #[test]
    fn invoke_nullary_macro() {
        let name = "my_macro";
        let actions = collect_semantic_actions(|actions| {
            let invocation = actions
                .enter_stmt(None)
                .enter_macro_invocation((name.into(), ()));
            invocation.exit().exit()
        });
        assert_eq!(
            actions,
            [TestOperation::InvokeMacro(name.into(), Vec::new())]
        )
    }

    #[test]
    fn invoke_unary_macro() {
        let name = "my_macro";
        let arg_token = Token::Literal(Literal::Operand(Operand::A));
        let actions = collect_semantic_actions(|actions| {
            let mut invocation = actions
                .enter_stmt(None)
                .enter_macro_invocation((name.into(), ()));
            invocation = {
                let mut arg = invocation.enter_macro_arg();
                arg.push_token((arg_token.clone(), ()));
                arg.exit()
            };
            invocation.exit().exit()
        });
        assert_eq!(
            actions,
            [TestOperation::InvokeMacro(
                name.into(),
                vec![vec![arg_token]]
            )]
        )
    }

    #[test]
    fn diagnose_wrong_operand_count() {
        let actions = collect_semantic_actions(|actions| {
            let mut arg = actions
                .enter_stmt(None)
                .enter_command((Command::Mnemonic(Mnemonic::Nop), ()))
                .add_argument();
            let literal_a = Literal::Operand(Operand::A);
            arg.push_atom((ExprAtom::Literal(literal_a), ()));
            arg.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [diag::Event::EmitDiagnostic(CompactDiagnostic::new(
                Message::OperandCount {
                    actual: 1,
                    expected: 0
                },
                ()
            ))
            .into()]
        )
    }

    #[test]
    fn diagnose_parsing_error() {
        let diagnostic = CompactDiagnostic::new(Message::UnexpectedToken { token: () }, ());
        let actions = collect_semantic_actions(|actions| {
            let mut stmt = actions.enter_stmt(None);
            stmt.diagnostics().emit_diagnostic(diagnostic.clone());
            stmt.exit()
        });
        assert_eq!(actions, [diag::Event::EmitDiagnostic(diagnostic).into()])
    }

    #[test]
    fn recover_from_malformed_expr() {
        let diagnostic = CompactDiagnostic::new(Message::UnexpectedToken { token: () }, ());
        let actions = collect_semantic_actions(|file| {
            let mut expr = file
                .enter_stmt(None)
                .enter_command((Command::Mnemonic(Mnemonic::Add), ()))
                .add_argument();
            expr.diagnostics().emit_diagnostic(diagnostic.clone());
            expr.exit().exit().exit()
        });
        assert_eq!(actions, [diag::Event::EmitDiagnostic(diagnostic).into()])
    }

    pub(crate) type MockBackend<'a> = backend::MockBackend<'a, TestOperation>;
    pub(super) type MockDiagnostics<'a> = diag::MockDiagnostics<'a, TestOperation, ()>;

    pub(crate) fn collect_semantic_actions<F>(f: F) -> Vec<TestOperation>
    where
        F: for<'a> FnOnce(TestSemanticActions<'a>) -> TestSemanticActions<'a>,
    {
        let operations = RefCell::new(Vec::new());
        let mut frontend = TestFrontend::new(&operations);
        let mut backend = MockBackend::new(&operations);
        let mut names = HashMapNameTable::new();
        let mut diagnostics = MockDiagnostics::new(&operations);
        let session = Session::new(&mut frontend, &mut backend, &mut names, &mut diagnostics);
        f(SemanticActions::new(session));
        operations.into_inner()
    }

    pub type TestNameTable<'a> = HashMapNameTable<MacroTableEntry<usize, Rc<MacroDefData<String>>>>;

    type TestSemanticActions<'a> = SemanticActions<
        'a,
        TestFrontend<'a>,
        MockBackend<'a>,
        TestNameTable<'a>,
        MockDiagnostics<'a>,
    >;
}
