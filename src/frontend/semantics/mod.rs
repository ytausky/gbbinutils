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
    pub enum SemanticAtom<I> {
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

    pub type SemanticExpr<I, S> = Expr<SemanticAtom<I>, SemanticUnary, BinaryOperator, S>;

    #[cfg(test)]
    pub type SemanticExprVariant<I, S> =
        ExprVariant<SemanticAtom<I>, SemanticUnary, BinaryOperator, S>;
}

use self::expr::*;

pub(crate) struct SemanticActions<'a, F: Frontend<D>, B, D: Diagnostics> {
    session: Session<'a, F, B, D>,
    label: Option<(Ident<F::StringRef>, D::Span)>,
}

impl<'a, F, B, D> SemanticActions<'a, F, B, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span>,
    D: Diagnostics,
{
    pub fn new(session: Session<'a, F, B, D>) -> SemanticActions<'a, F, B, D> {
        SemanticActions {
            session,
            label: None,
        }
    }

    fn define_label_if_present(&mut self) {
        if let Some((label, span)) = self.label.take() {
            let value = {
                let mut builder = self.session.backend.build_value();
                ToValue::<LocationCounter, D::Span>::to_value(
                    &mut builder,
                    (LocationCounter, span.clone()),
                )
            };
            self.session.backend.define_symbol((label, span), value)
        }
    }
}

impl<'a, F, B, D> DelegateDiagnostics<D::Span> for SemanticActions<'a, F, B, D>
where
    F: Frontend<D>,
    D: Diagnostics,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.session.diagnostics
    }
}

impl<'a, F, B, D> syntax::FileContext<Ident<F::StringRef>, Literal<F::StringRef>, Command, D::Span>
    for SemanticActions<'a, F, B, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span>,
    D: Diagnostics,
{
    type StmtContext = Self;

    fn enter_stmt(mut self, label: Option<(Ident<F::StringRef>, D::Span)>) -> Self::StmtContext {
        self.label = label;
        self
    }
}

impl<'a, F, B, D> syntax::StmtContext<Ident<F::StringRef>, Literal<F::StringRef>, Command, D::Span>
    for SemanticActions<'a, F, B, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span>,
    D: Diagnostics,
{
    type CommandContext = CommandActions<'a, F, B, D>;
    type MacroParamsContext = MacroDefActions<'a, F, B, D>;
    type MacroInvocationContext = MacroInvocationActions<'a, F, B, D>;
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

pub(crate) struct CommandActions<'a, F: Frontend<D>, B, D: Diagnostics> {
    name: (Command, D::Span),
    args: CommandArgs<Ident<F::StringRef>, D::Span>,
    parent: SemanticActions<'a, F, B, D>,
    has_errors: bool,
}

type CommandArgs<I, S> = Vec<SemanticExpr<I, S>>;

impl<'a, F: Frontend<D>, B, D: Diagnostics> CommandActions<'a, F, B, D> {
    fn new(
        name: (Command, D::Span),
        parent: SemanticActions<'a, F, B, D>,
    ) -> CommandActions<'a, F, B, D> {
        CommandActions {
            name,
            args: Vec::new(),
            parent,
            has_errors: false,
        }
    }
}

impl<'a, F: Frontend<D>, B, D: Diagnostics> MergeSpans<D::Span> for CommandActions<'a, F, B, D> {
    fn merge_spans(&mut self, left: &D::Span, right: &D::Span) -> D::Span {
        self.parent.diagnostics().merge_spans(left, right)
    }
}

impl<'a, F: Frontend<D>, B, D: Diagnostics> StripSpan<D::Span> for CommandActions<'a, F, B, D> {
    type Stripped = D::Stripped;

    fn strip_span(&mut self, span: &D::Span) -> Self::Stripped {
        self.parent.diagnostics().strip_span(span)
    }
}

impl<'a, F, B, D> EmitDiagnostic<D::Span, D::Stripped> for CommandActions<'a, F, B, D>
where
    F: Frontend<D>,
    D: Diagnostics,
{
    fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<D::Span, D::Stripped>) {
        self.has_errors = true;
        self.parent.diagnostics().emit_diagnostic(diagnostic)
    }
}

impl<'a, F, B, D> DelegateDiagnostics<D::Span> for CommandActions<'a, F, B, D>
where
    F: Frontend<D>,
    D: Diagnostics,
{
    type Delegate = Self;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self
    }
}

impl<'a, F, B, D> syntax::CommandContext<D::Span> for CommandActions<'a, F, B, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span>,
    D: Diagnostics,
{
    type Ident = Ident<F::StringRef>;
    type Command = Command;
    type Literal = Literal<F::StringRef>;
    type ArgContext = ExprContext<'a, F, B, D>;
    type Parent = SemanticActions<'a, F, B, D>;

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

pub(crate) struct ExprContext<'a, F: Frontend<D>, B, D: Diagnostics> {
    stack: Vec<SemanticExpr<F::StringRef, D::Span>>,
    parent: CommandActions<'a, F, B, D>,
}

impl<'a, F, B, D> DelegateDiagnostics<D::Span> for ExprContext<'a, F, B, D>
where
    F: Frontend<D>,
    D: Diagnostics,
{
    type Delegate = CommandActions<'a, F, B, D>;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        &mut self.parent
    }
}

impl<'a, F, B, D> syntax::ExprContext<D::Span> for ExprContext<'a, F, B, D>
where
    F: Frontend<D>,
    D: Diagnostics,
{
    type Ident = Ident<F::StringRef>;
    type Literal = Literal<F::StringRef>;
    type Parent = CommandActions<'a, F, B, D>;

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

fn analyze_mnemonic<'a, F, B, D>(
    name: (Mnemonic, D::Span),
    args: CommandArgs<F::StringRef, D::Span>,
    actions: &mut SemanticActions<'a, F, B, D>,
) where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span>,
    D: Diagnostics,
{
    let result = instruction::analyze_instruction(
        name,
        args.into_iter(),
        ValueContext::new(
            &mut actions.session.backend.build_value(),
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

pub(crate) struct MacroDefActions<'a, F: Frontend<D>, B, D: Diagnostics> {
    name: Option<(Ident<F::StringRef>, D::Span)>,
    params: Vec<(Ident<F::StringRef>, D::Span)>,
    tokens: Vec<(SemanticToken<F::StringRef>, D::Span)>,
    parent: SemanticActions<'a, F, B, D>,
}

impl<'a, F: Frontend<D>, B, D: Diagnostics> MacroDefActions<'a, F, B, D> {
    fn new(
        name: Option<(Ident<F::StringRef>, D::Span)>,
        parent: SemanticActions<'a, F, B, D>,
    ) -> MacroDefActions<'a, F, B, D> {
        MacroDefActions {
            name,
            params: Vec::new(),
            tokens: Vec::new(),
            parent,
        }
    }
}

impl<'a, F, B, D> DelegateDiagnostics<D::Span> for MacroDefActions<'a, F, B, D>
where
    F: Frontend<D>,
    D: Diagnostics,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.diagnostics()
    }
}

impl<'a, F, B, D> syntax::MacroParamsContext<D::Span> for MacroDefActions<'a, F, B, D>
where
    F: Frontend<D>,
    D: Diagnostics,
{
    type Ident = Ident<F::StringRef>;
    type Command = Command;
    type Literal = Literal<F::StringRef>;
    type MacroBodyContext = Self;
    type Parent = SemanticActions<'a, F, B, D>;

    fn add_parameter(&mut self, param: (Self::Ident, D::Span)) {
        self.params.push(param)
    }

    fn exit(self) -> Self::MacroBodyContext {
        self
    }
}

impl<'a, F, B, D> syntax::TokenSeqContext<D::Span> for MacroDefActions<'a, F, B, D>
where
    F: Frontend<D>,
    D: Diagnostics,
{
    type Token = SemanticToken<F::StringRef>;
    type Parent = SemanticActions<'a, F, B, D>;

    fn push_token(&mut self, token: (Self::Token, D::Span)) {
        self.tokens.push(token)
    }

    fn exit(mut self) -> Self::Parent {
        if let Some(name) = self.name {
            self.parent
                .session
                .define_macro(name, self.params, self.tokens)
        }
        self.parent
    }
}

pub(crate) struct MacroInvocationActions<'a, F: Frontend<D>, B, D: Diagnostics> {
    name: (Ident<F::StringRef>, D::Span),
    args: Vec<super::TokenSeq<F::StringRef, D::Span>>,
    parent: SemanticActions<'a, F, B, D>,
}

impl<'a, F: Frontend<D>, B, D: Diagnostics> MacroInvocationActions<'a, F, B, D> {
    fn new(
        name: (Ident<F::StringRef>, D::Span),
        parent: SemanticActions<'a, F, B, D>,
    ) -> MacroInvocationActions<'a, F, B, D> {
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

impl<'a, F, B, D> DelegateDiagnostics<D::Span> for MacroInvocationActions<'a, F, B, D>
where
    F: Frontend<D>,
    D: Diagnostics,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.diagnostics()
    }
}

impl<'a, F, B, D> syntax::MacroInvocationContext<D::Span> for MacroInvocationActions<'a, F, B, D>
where
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span>,
    D: Diagnostics,
{
    type Token = SemanticToken<F::StringRef>;
    type Parent = SemanticActions<'a, F, B, D>;
    type MacroArgContext = MacroArgContext<'a, F, B, D>;

    fn enter_macro_arg(self) -> Self::MacroArgContext {
        MacroArgContext::new(self)
    }

    fn exit(mut self) -> Self::Parent {
        self.parent.session.invoke_macro(self.name, self.args);
        self.parent
    }
}

pub(crate) struct MacroArgContext<'a, F: Frontend<D>, B, D: Diagnostics> {
    tokens: Vec<(SemanticToken<F::StringRef>, D::Span)>,
    parent: MacroInvocationActions<'a, F, B, D>,
}

impl<'a, F: Frontend<D>, B, D: Diagnostics> MacroArgContext<'a, F, B, D> {
    fn new(parent: MacroInvocationActions<'a, F, B, D>) -> MacroArgContext<'a, F, B, D> {
        MacroArgContext {
            tokens: Vec::new(),
            parent,
        }
    }
}

impl<'a, F, B, D> DelegateDiagnostics<D::Span> for MacroArgContext<'a, F, B, D>
where
    F: Frontend<D>,
    D: Diagnostics,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.parent.diagnostics()
    }
}

impl<'a, F, B, D> syntax::TokenSeqContext<D::Span> for MacroArgContext<'a, F, B, D>
where
    F: Frontend<D>,
    D: Diagnostics,
{
    type Token = SemanticToken<F::StringRef>;
    type Parent = MacroInvocationActions<'a, F, B, D>;

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
    B: ValueBuilder<I, S>,
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

    use crate::backend::{
        BuildValue, HasValue, IndependentValueBuilder, RelocAtom, RelocExpr, Width,
    };
    use crate::codebase::{BufId, BufRange, CodebaseError};
    use crate::diag::{CompactDiagnostic, Message};
    use crate::expr::BinaryOperator;
    use crate::frontend::syntax::{
        keyword::Operand, CommandContext, ExprContext, FileContext, MacroInvocationContext,
        MacroParamsContext, StmtContext, Token, TokenSeqContext,
    };
    use crate::frontend::{Downstream, MacroArgs};
    use crate::span::*;
    use std::borrow::Borrow;
    use std::cell::RefCell;

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

    impl<'a> Frontend<TestDiagnostics<'a>> for TestFrontend<'a> {
        type StringRef = String;
        type MacroDefId = usize;

        fn analyze_file<B>(
            &mut self,
            path: Self::StringRef,
            _downstream: Downstream<B, TestDiagnostics<'a>>,
        ) -> Result<(), CodebaseError>
        where
            B: Backend<String, ()>,
        {
            self.operations
                .borrow_mut()
                .push(TestOperation::AnalyzeFile(path));
            match self.error.take() {
                Some(error) => Err(error),
                None => Ok(()),
            }
        }

        fn invoke_macro<B>(
            &mut self,
            name: (Ident<Self::StringRef>, ()),
            args: MacroArgs<Self::StringRef, ()>,
            _downstream: Downstream<B, TestDiagnostics<'a>>,
        ) where
            B: Backend<String, ()>,
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
            name: (impl Into<Ident<Self::StringRef>>, ()),
            params: Vec<(Ident<Self::StringRef>, ())>,
            tokens: Vec<(SemanticToken<Self::StringRef>, ())>,
            _diagnostics: &mut TestDiagnostics<'a>,
        ) {
            self.operations
                .borrow_mut()
                .push(TestOperation::DefineMacro(
                    name.0.into(),
                    params.into_iter().map(|(s, _)| s).collect(),
                    tokens.into_iter().map(|(t, _)| t).collect(),
                ))
        }
    }

    pub(crate) struct TestBackend<'a> {
        operations: &'a RefCell<Vec<TestOperation>>,
    }

    impl<'a> TestBackend<'a> {
        pub fn new(operations: &'a RefCell<Vec<TestOperation>>) -> Self {
            TestBackend { operations }
        }
    }

    impl<'a> HasValue<()> for TestBackend<'a> {
        type Value = RelocExpr<String, ()>;
    }

    impl<'a, 'b> BuildValue<'b, String, ()> for TestBackend<'a> {
        type Builder = IndependentValueBuilder<()>;

        fn build_value(&mut self) -> Self::Builder {
            IndependentValueBuilder::new()
        }
    }

    impl<'a> Backend<String, ()> for TestBackend<'a> {
        type Object = ();

        fn define_symbol(&mut self, symbol: (String, ()), value: RelocExpr<String, ()>) {
            self.operations
                .borrow_mut()
                .push(TestOperation::DefineSymbol(symbol.0, value))
        }

        fn emit_item(&mut self, item: backend::Item<RelocExpr<String, ()>>) {
            self.operations
                .borrow_mut()
                .push(TestOperation::EmitItem(item))
        }

        fn into_object(self) -> Self::Object {}

        fn set_origin(&mut self, origin: RelocExpr<String, ()>) {
            self.operations
                .borrow_mut()
                .push(TestOperation::SetOrigin(origin))
        }
    }

    pub(crate) struct TestDiagnostics<'a> {
        operations: &'a RefCell<Vec<TestOperation>>,
    }

    impl<'a> TestDiagnostics<'a> {
        pub fn new(operations: &'a RefCell<Vec<TestOperation>>) -> Self {
            TestDiagnostics { operations }
        }
    }

    impl<'a> Diagnostics for TestDiagnostics<'a> {}

    impl<'a> Span for TestDiagnostics<'a> {
        type Span = ();
    }

    impl<'a> MergeSpans<()> for TestDiagnostics<'a> {
        fn merge_spans(&mut self, _: &(), _: &()) {}
    }

    impl<'a> StripSpan<()> for TestDiagnostics<'a> {
        type Stripped = ();

        fn strip_span(&mut self, _: &()) {}
    }

    impl<'a> EmitDiagnostic<(), ()> for TestDiagnostics<'a> {
        fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<(), ()>) {
            self.operations
                .borrow_mut()
                .push(TestOperation::EmitDiagnostic(diagnostic))
        }
    }

    impl<'a> MacroContextFactory<()> for TestDiagnostics<'a> {
        type MacroDefId = usize;
        type MacroExpansionContext = TestMacroExpansionContext;

        fn add_macro_def<P, B>(&mut self, _name: (), _params: P, _body: B) -> Self::MacroDefId
        where
            P: IntoIterator<Item = ()>,
            B: IntoIterator<Item = ()>,
        {
            0
        }

        fn mk_macro_expansion_context<A, J>(
            &mut self,
            _name: (),
            _args: A,
            _def: &Self::MacroDefId,
        ) -> Self::MacroExpansionContext
        where
            A: IntoIterator<Item = J>,
            J: IntoIterator<Item = ()>,
        {
            TestMacroExpansionContext
        }
    }

    pub struct TestMacroExpansionContext;

    impl MacroExpansionContext for TestMacroExpansionContext {
        type Span = ();

        fn mk_span(&self, _token: usize, _expansion: Option<TokenExpansion>) -> Self::Span {}
    }

    impl<'a> ContextFactory for TestDiagnostics<'a> {
        type BufContext = TestBufContext;

        fn mk_buf_context(
            &mut self,
            _buf_id: BufId,
            _included_from: Option<Self::Span>,
        ) -> Self::BufContext {
            TestBufContext
        }
    }

    pub struct TestBufContext;

    impl BufContext for TestBufContext {
        type Span = ();

        fn mk_span(&self, _range: BufRange) -> Self::Span {}
    }

    #[derive(Debug, PartialEq)]
    pub(crate) enum TestOperation {
        AnalyzeFile(String),
        InvokeMacro(String, Vec<Vec<SemanticToken<String>>>),
        DefineMacro(String, Vec<String>, Vec<SemanticToken<String>>),
        DefineSymbol(String, RelocExpr<String, ()>),
        EmitDiagnostic(CompactDiagnostic<(), ()>),
        EmitItem(backend::Item<RelocExpr<String, ()>>),
        SetOrigin(RelocExpr<String, ()>),
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
            [TestOperation::EmitItem(backend::Item::Instruction(
                Instruction::Ld(Ld::Simple(SimpleOperand::B, SimpleOperand::DerefHl))
            ))]
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
            [TestOperation::EmitItem(backend::Item::Instruction(
                Instruction::Rst(
                    ExprVariant::Binary(op, Box::new(1.into()), Box::new(1.into()),).into()
                )
            ))]
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
            arg.push_atom((ExprAtom::Ident(label.to_string()), ()));
            arg.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [TestOperation::EmitItem(backend::Item::Data(
                RelocAtom::Symbol(label.to_string()).into(),
                Width::Word
            ))]
        );
    }

    #[test]
    fn analyze_label() {
        let label = "label";
        let actions = collect_semantic_actions(|actions| {
            actions.enter_stmt(Some((label.to_string(), ()))).exit()
        });
        assert_eq!(
            actions,
            [TestOperation::DefineSymbol(
                label.to_string(),
                RelocAtom::LocationCounter.into()
            )]
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
                Token::Ident(param.to_string()),
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
            [TestOperation::EmitDiagnostic(CompactDiagnostic::new(
                Message::MacroRequiresName,
                ()
            ))]
        )
    }

    fn test_macro_definition(
        name: &str,
        params: impl Borrow<[&'static str]>,
        body: impl Borrow<[SemanticToken<String>]>,
    ) {
        let actions = collect_semantic_actions(|actions| {
            let mut params_actions = actions
                .enter_stmt(Some((name.to_string(), ())))
                .enter_macro_def(());
            for param in params.borrow().iter().map(|t| (t.to_string(), ())) {
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
                name.to_string(),
                params.borrow().iter().cloned().map(String::from).collect(),
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
                .enter_macro_invocation((name.to_string(), ()));
            invocation.exit().exit()
        });
        assert_eq!(
            actions,
            [TestOperation::InvokeMacro(name.to_string(), Vec::new())]
        )
    }

    #[test]
    fn invoke_unary_macro() {
        let name = "my_macro";
        let arg_token = Token::Literal(Literal::Operand(Operand::A));
        let actions = collect_semantic_actions(|actions| {
            let mut invocation = actions
                .enter_stmt(None)
                .enter_macro_invocation((name.to_string(), ()));
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
                name.to_string(),
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
            [TestOperation::EmitDiagnostic(CompactDiagnostic::new(
                Message::OperandCount {
                    actual: 1,
                    expected: 0
                },
                ()
            ))]
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
        assert_eq!(actions, [TestOperation::EmitDiagnostic(diagnostic)])
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
        assert_eq!(actions, [TestOperation::EmitDiagnostic(diagnostic)])
    }

    pub(crate) fn collect_semantic_actions<F>(f: F) -> Vec<TestOperation>
    where
        F: for<'a> FnOnce(TestSemanticActions<'a>) -> TestSemanticActions<'a>,
    {
        let operations = RefCell::new(Vec::new());
        let mut frontend = TestFrontend::new(&operations);
        let mut backend = TestBackend::new(&operations);
        let mut diagnostics = TestDiagnostics::new(&operations);
        let session = Session::new(&mut frontend, &mut backend, &mut diagnostics);
        f(SemanticActions::new(session));
        operations.into_inner()
    }

    type TestSemanticActions<'a> =
        SemanticActions<'a, TestFrontend<'a>, TestBackend<'a>, TestDiagnostics<'a>>;
}
