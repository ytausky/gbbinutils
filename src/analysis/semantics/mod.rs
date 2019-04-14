use self::invoke::MacroInvocationActions;

use super::backend::{Backend, LocationCounter, PushOp};
use super::macros::MacroEntry;
use super::session::*;
use super::{Ident, Lex, LexItem, Literal, SemanticToken};

use crate::diag::span::{MergeSpans, Source, StripSpan};
use crate::diag::*;
use crate::model::{BinOp, Item};
use crate::name::{NameTable, StartScope};
use crate::syntax::{self, keyword::*, ExprAtom, Operator, UnaryOperator};

use std::marker::PhantomData;

#[cfg(test)]
pub use self::mock::*;

mod directive;
mod instruction;
mod invoke;
mod operand;

#[derive(Clone, Debug, PartialEq)]
enum SemanticAtom<I> {
    Ident(Ident<I>),
    Literal(Literal<I>),
    LocationCounter,
}

impl<I> From<Literal<I>> for SemanticAtom<I> {
    fn from(literal: Literal<I>) -> Self {
        SemanticAtom::Literal(literal)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum SemanticUnary {
    Parentheses,
}

#[derive(Clone, Debug, PartialEq)]
struct SemanticExpr<I, S> {
    pub variant: ExprVariant<I, S>,
    pub span: S,
}

#[derive(Clone, Debug, PartialEq)]
enum ExprVariant<I, S> {
    Atom(SemanticAtom<I>),
    Unary(SemanticUnary, Box<SemanticExpr<I, S>>),
    Binary(BinOp, Box<SemanticExpr<I, S>>, Box<SemanticExpr<I, S>>),
}

#[cfg(test)]
impl<I, S> SemanticExpr<I, S> {
    pub fn from_atom<T: Into<ExprVariant<I, S>>>(atom: T, span: S) -> Self {
        Self {
            variant: atom.into(),
            span,
        }
    }
}

impl<I, S> From<SemanticAtom<I>> for ExprVariant<I, S> {
    fn from(atom: SemanticAtom<I>) -> Self {
        ExprVariant::Atom(atom)
    }
}

impl<I, S: Clone> Source for SemanticExpr<I, S> {
    type Span = S;

    fn span(&self) -> Self::Span {
        self.span.clone()
    }
}

pub(crate) trait Analyze<R: Clone + Eq, D: Diagnostics> {
    fn analyze_token_seq<I, C, B, N>(
        &mut self,
        tokens: I,
        partial: &mut PartialSession<C, B, N, D>,
    ) where
        I: IntoIterator<Item = LexItem<R, D::Span>>,
        C: Lex<D, StringRef = R>,
        B: Backend<D::Span> + ?Sized,
        N: NameTable<Ident<R>, BackendEntry = B::Name, MacroEntry = MacroEntry<R, D>>
            + StartScope<Ident<R>>,
        B::Value: Default + ValueBuilder<B::Name, D::Span>;
}

pub struct SemanticAnalyzer;

impl<R: Clone + Eq, D: Diagnostics> Analyze<R, D> for SemanticAnalyzer {
    fn analyze_token_seq<I, C, B, N>(&mut self, tokens: I, partial: &mut PartialSession<C, B, N, D>)
    where
        I: IntoIterator<Item = LexItem<R, D::Span>>,
        C: Lex<D, StringRef = R>,
        B: Backend<D::Span> + ?Sized,
        N: NameTable<Ident<R>, BackendEntry = B::Name, MacroEntry = MacroEntry<R, D>>
            + StartScope<Ident<R>>,
        B::Value: Default + ValueBuilder<B::Name, D::Span>,
    {
        let session = CompositeSession::new(
            partial.codebase,
            self,
            partial.backend,
            partial.names,
            partial.diagnostics,
        );
        let actions = SemanticActions::new(session);
        crate::syntax::parse_token_seq(tokens.into_iter(), actions);
    }
}

pub(crate) struct SemanticActions<S: Session> {
    session: Option<S>,
    label: Option<(Ident<S::StringRef>, S::Span)>,
}

impl<S: Session> SemanticActions<S> {
    pub fn new(session: S) -> SemanticActions<S> {
        SemanticActions {
            session: Some(session),
            label: None,
        }
    }

    fn analyze_expr(&mut self, expr: SemanticExpr<S::StringRef, S::Span>) -> Result<S::Value, ()> {
        self.build_value(|mut builder| {
            let result = builder.analyze_expr(expr);
            let (session, value) = builder.finish();
            (session, result.map(|()| value))
        })
    }

    fn build_value<F, T>(&mut self, f: F) -> T
    where
        F: FnOnce(S::GeneralBuilder) -> (S, T),
    {
        let builder = self.session.take().unwrap().build_value();
        let result = f(builder);
        self.session = Some(result.0);
        result.1
    }

    fn define_label_if_present(&mut self) {
        if let Some((label, span)) = self.label.take() {
            let value = self.build_value(|mut builder| {
                PushOp::<LocationCounter, _>::push_op(&mut builder, LocationCounter, span.clone());
                builder.finish()
            });
            self.session().define_symbol((label, span), value)
        }
    }

    fn session(&mut self) -> &mut S {
        self.session.as_mut().unwrap()
    }
}

impl<S: Session> DelegateDiagnostics<S::Span> for SemanticActions<S> {
    type Delegate = S::Delegate;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.session().diagnostics()
    }
}

impl<S: Session> syntax::FileContext<Ident<S::StringRef>, Literal<S::StringRef>, Command, S::Span>
    for SemanticActions<S>
{
    type StmtContext = Self;

    fn enter_stmt(mut self, label: Option<(Ident<S::StringRef>, S::Span)>) -> Self::StmtContext {
        self.label = label;
        self
    }
}

impl<S: Session> syntax::StmtContext<Ident<S::StringRef>, Literal<S::StringRef>, Command, S::Span>
    for SemanticActions<S>
{
    type CommandContext = CommandActions<S>;
    type ExprParamsContext = DefHeadActions<S, ExprDef>;
    type MacroParamsContext = DefHeadActions<S, MacroDef>;
    type MacroInvocationContext = MacroInvocationActions<S>;
    type Parent = Self;

    fn enter_command(self, name: (Command, S::Span)) -> Self::CommandContext {
        CommandActions::new(name, self)
    }

    fn enter_expr_def(mut self, _keyword: S::Span) -> Self::ExprParamsContext {
        DefHeadActions::new(self.label.take(), self)
    }

    fn enter_macro_def(mut self, keyword: S::Span) -> Self::MacroParamsContext {
        if self.label.is_none() {
            self.diagnostics()
                .emit_diagnostic(Message::MacroRequiresName.at(keyword))
        }
        DefHeadActions::new(self.label.take(), self)
    }

    fn enter_macro_invocation(
        mut self,
        name: (Ident<S::StringRef>, S::Span),
    ) -> Self::MacroInvocationContext {
        self.define_label_if_present();
        MacroInvocationActions::new(name, self)
    }

    fn exit(mut self) -> Self::Parent {
        self.define_label_if_present();
        self
    }
}

pub(crate) struct CommandActions<S: Session> {
    name: (Command, S::Span),
    args: CommandArgs<S::StringRef, S::Span>,
    parent: SemanticActions<S>,
    has_errors: bool,
}

type CommandArgs<I, S> = Vec<SemanticExpr<I, S>>;

impl<S: Session> CommandActions<S> {
    fn new(name: (Command, S::Span), parent: SemanticActions<S>) -> CommandActions<S> {
        CommandActions {
            name,
            args: Vec::new(),
            parent,
            has_errors: false,
        }
    }
}

impl<S: Session> MergeSpans<S::Span> for CommandActions<S> {
    fn merge_spans(&mut self, left: &S::Span, right: &S::Span) -> S::Span {
        self.parent.diagnostics().merge_spans(left, right)
    }
}

impl<S: Session> StripSpan<S::Span> for CommandActions<S> {
    type Stripped = <S::Delegate as StripSpan<S::Span>>::Stripped;

    fn strip_span(&mut self, span: &S::Span) -> Self::Stripped {
        self.parent.diagnostics().strip_span(span)
    }
}

impl<S: Session> EmitDiagnostic<S::Span, <S::Delegate as StripSpan<S::Span>>::Stripped>
    for CommandActions<S>
{
    fn emit_diagnostic(
        &mut self,
        diagnostic: impl Into<CompactDiagnostic<S::Span, <S::Delegate as StripSpan<S::Span>>::Stripped>>,
    ) {
        self.has_errors = true;
        self.parent.diagnostics().emit_diagnostic(diagnostic)
    }
}

impl<S: Session> DelegateDiagnostics<S::Span> for CommandActions<S> {
    type Delegate = Self;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self
    }
}

impl<S: Session> syntax::CommandContext<S::Span> for CommandActions<S> {
    type Ident = Ident<S::StringRef>;
    type Command = Command;
    type Literal = Literal<S::StringRef>;
    type ArgContext = ExprContext<S::StringRef, S::Span, Self>;
    type Parent = SemanticActions<S>;

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
            Directive::Equ | Directive::Section => true,
            _ => false,
        }
    }
}

pub(crate) struct ExprContext<R, S, P> {
    stack: Vec<SemanticExpr<R, S>>,
    parent: P,
}

impl<R, S, P> ExprContext<R, S, P> {
    fn pop(&mut self) -> SemanticExpr<R, S> {
        self.stack.pop().unwrap_or_else(|| unreachable!())
    }
}

impl<R, S, P> DelegateDiagnostics<S> for ExprContext<R, S, P>
where
    P: DelegateDiagnostics<S>,
{
    type Delegate = P::Delegate;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.diagnostics()
    }
}

impl<R, S, P> syntax::AssocIdent for ExprContext<R, S, P> {
    type Ident = Ident<R>;
}

impl<S: Session> syntax::FinalContext for ExprContext<S::StringRef, S::Span, CommandActions<S>> {
    type ReturnTo = CommandActions<S>;

    fn exit(mut self) -> Self::ReturnTo {
        if !self.parent.has_errors {
            assert_eq!(self.stack.len(), 1);
            self.parent.args.push(self.stack.pop().unwrap());
        }
        self.parent
    }
}

impl<R, S, P> syntax::ExprContext<S> for ExprContext<R, S, P>
where
    S: Clone,
    Self: DelegateDiagnostics<S>,
{
    type Literal = Literal<R>;

    fn push_atom(&mut self, atom: (ExprAtom<Self::Ident, Self::Literal>, S)) {
        self.stack.push(SemanticExpr {
            variant: ExprVariant::Atom(match atom.0 {
                ExprAtom::Ident(ident) => SemanticAtom::Ident(ident),
                ExprAtom::Literal(literal) => SemanticAtom::Literal(literal),
                ExprAtom::LocationCounter => SemanticAtom::LocationCounter,
            }),
            span: atom.1,
        })
    }

    fn apply_operator(&mut self, operator: (Operator, S)) {
        match operator.0 {
            Operator::Unary(UnaryOperator::Parentheses) => {
                let inner = self.pop();
                self.stack.push(SemanticExpr {
                    variant: ExprVariant::Unary(SemanticUnary::Parentheses, Box::new(inner)),
                    span: operator.1,
                })
            }
            Operator::Binary(binary) => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.stack.push(SemanticExpr {
                    variant: ExprVariant::Binary(binary, Box::new(lhs), Box::new(rhs)),
                    span: operator.1,
                })
            }
            Operator::FnCall(_) => unimplemented!(),
        }
    }
}

fn analyze_mnemonic<S: Session>(
    name: (Mnemonic, S::Span),
    args: CommandArgs<S::StringRef, S::Span>,
    actions: &mut SemanticActions<S>,
) {
    let operands: Vec<_> = args
        .into_iter()
        .map(|arg| {
            actions.build_value(|builder| operand::analyze_operand(arg, name.0.context(), builder))
        })
        .collect();
    if let Ok(instruction) = instruction::analyze_instruction(name, operands, actions.diagnostics())
    {
        actions.session().emit_item(Item::Instruction(instruction))
    }
}

pub(crate) struct DefHeadActions<S: Session, T> {
    name: Option<(Ident<S::StringRef>, S::Span)>,
    params: (Vec<Ident<S::StringRef>>, Vec<S::Span>),
    actions: SemanticActions<S>,
    _tag: PhantomData<T>,
}

impl<S: Session, T> DefHeadActions<S, T> {
    fn new(name: Option<(Ident<S::StringRef>, S::Span)>, actions: SemanticActions<S>) -> Self {
        Self {
            name,
            params: (Vec::new(), Vec::new()),
            actions,
            _tag: PhantomData,
        }
    }
}

impl<S: Session, T> syntax::AssocIdent for DefHeadActions<S, T> {
    type Ident = Ident<S::StringRef>;
}

impl<S: Session, T> DelegateDiagnostics<S::Span> for DefHeadActions<S, T> {
    type Delegate = S::Delegate;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.actions.diagnostics()
    }
}

impl<S: Session, T> syntax::ParamsContext<S::Span> for DefHeadActions<S, T> {
    fn add_parameter(&mut self, (param, span): (Self::Ident, S::Span)) {
        self.params.0.push(param);
        self.params.1.push(span)
    }
}

pub(crate) struct ExprDef;

impl<S: Session> syntax::ToExprBody<S::Span> for DefHeadActions<S, ExprDef> {
    type Literal = Literal<S::StringRef>;
    type Parent = SemanticActions<S>;
    type Next = ExprContext<S::StringRef, S::Span, Self>;

    fn next(self) -> Self::Next {
        ExprContext {
            stack: Vec::new(),
            parent: self,
        }
    }
}

impl<S: Session> syntax::FinalContext
    for ExprContext<S::StringRef, S::Span, DefHeadActions<S, ExprDef>>
{
    type ReturnTo = SemanticActions<S>;

    fn exit(mut self) -> Self::ReturnTo {
        let mut builder = self
            .parent
            .actions
            .session
            .unwrap()
            .define_fn(self.parent.name.unwrap());
        builder.analyze_expr(self.stack.pop().unwrap()).unwrap();
        self.parent.actions.session = Some(builder.finish_fn_def());
        self.parent.actions
    }
}

pub(crate) struct MacroDef;

impl<S: Session> syntax::ToMacroBody<S::Span> for DefHeadActions<S, MacroDef> {
    type Literal = Literal<S::StringRef>;
    type Command = Command;
    type Parent = SemanticActions<S>;
    type Next = MacroDefActions<S>;

    fn next(self) -> Self::Next {
        MacroDefActions::new(self)
    }
}

pub(crate) struct MacroDefActions<S: Session> {
    head: DefHeadActions<S, MacroDef>,
    tokens: (Vec<SemanticToken<S::StringRef>>, Vec<S::Span>),
}

impl<S: Session> MacroDefActions<S> {
    fn new(head: DefHeadActions<S, MacroDef>) -> Self {
        Self {
            head,
            tokens: (Vec::new(), Vec::new()),
        }
    }
}

impl<S: Session> DelegateDiagnostics<S::Span> for MacroDefActions<S> {
    type Delegate = S::Delegate;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.head.diagnostics()
    }
}

impl<S: Session> syntax::TokenSeqContext<S::Span> for MacroDefActions<S> {
    type Token = SemanticToken<S::StringRef>;
    type Parent = SemanticActions<S>;

    fn push_token(&mut self, (token, span): (Self::Token, S::Span)) {
        self.tokens.0.push(token);
        self.tokens.1.push(span)
    }

    fn exit(self) -> Self::Parent {
        let mut actions = self.head.actions;
        if let Some(name) = self.head.name {
            actions
                .session
                .as_mut()
                .unwrap()
                .define_macro(name, self.head.params, self.tokens)
        }
        actions
    }
}

trait AnalyzeExpr<I, S: Clone> {
    fn analyze_expr(&mut self, expr: SemanticExpr<I, S>) -> Result<(), ()>;
}

impl<'a, T, I, S> AnalyzeExpr<I, S> for T
where
    T: ValueBuilder<Ident<I>, S> + DelegateDiagnostics<S>,
    S: Clone,
{
    fn analyze_expr(&mut self, expr: SemanticExpr<I, S>) -> Result<(), ()> {
        match expr.variant {
            ExprVariant::Atom(SemanticAtom::Ident(ident)) => {
                self.push_op(ident, expr.span);
                Ok(())
            }
            ExprVariant::Atom(SemanticAtom::Literal(Literal::Number(n))) => {
                self.push_op(n, expr.span);
                Ok(())
            }
            ExprVariant::Atom(SemanticAtom::Literal(Literal::Operand(_))) => {
                Err(Message::KeywordInExpr {
                    keyword: self.diagnostics().strip_span(&expr.span),
                }
                .at(expr.span))
            }
            ExprVariant::Atom(SemanticAtom::Literal(Literal::String(_))) => {
                Err(Message::StringInInstruction.at(expr.span))
            }
            ExprVariant::Atom(SemanticAtom::LocationCounter) => {
                self.push_op(LocationCounter, expr.span);
                Ok(())
            }
            ExprVariant::Unary(SemanticUnary::Parentheses, expr) => Ok(self.analyze_expr(*expr)?),
            ExprVariant::Binary(binary, left, right) => {
                self.analyze_expr(*left)?;
                self.analyze_expr(*right)?;
                self.push_op(binary, expr.span);
                Ok(())
            }
        }
        .map_err(|diagnostic| {
            self.diagnostics().emit_diagnostic(diagnostic);
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
impl MockSpan for TokenSpan {
    fn default() -> Self {
        unimplemented!()
    }

    fn merge(&self, other: &Self) -> Self {
        TokenSpan::merge(self, other)
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use std::cell::RefCell;

    pub struct MockAnalyzer<'a, T> {
        log: &'a RefCell<Vec<T>>,
    }

    impl<'a, T> MockAnalyzer<'a, T> {
        pub fn new(log: &'a RefCell<Vec<T>>) -> Self {
            Self { log }
        }
    }

    impl<'a, T, D> Analyze<String, D> for MockAnalyzer<'a, T>
    where
        T: From<AnalyzerEvent<D::Span>>,
        D: Diagnostics,
    {
        fn analyze_token_seq<I, C, B, N>(
            &mut self,
            tokens: I,
            _downstream: &mut PartialSession<C, B, N, D>,
        ) where
            I: IntoIterator<Item = LexItem<String, D::Span>>,
            C: Lex<D, StringRef = String>,
            B: Backend<D::Span> + ?Sized,
            N: NameTable<Ident<String>, MacroEntry = MacroEntry<String, D>>,
            B::Value: Default + ValueBuilder<B::Name, D::Span>,
        {
            self.log
                .borrow_mut()
                .push(AnalyzerEvent::AnalyzeTokenSeq(tokens.into_iter().collect()).into())
        }
    }

    #[derive(Debug, PartialEq)]
    pub(crate) enum AnalyzerEvent<S> {
        AnalyzeTokenSeq(Vec<LexItem<String, S>>),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::backend::BackendEvent;
    use crate::analysis::session::SessionEvent;
    use crate::diag::Message;
    use crate::model::{Atom, BinOp, Width};
    use crate::syntax::{
        CommandContext, ExprContext, FileContext, FinalContext, MacroInvocationContext,
        ParamsContext, StmtContext, ToExprBody, ToMacroBody, TokenSeqContext,
    };
    use crate::syntax::{Operand, Token};

    use std::borrow::Borrow;
    use std::cell::RefCell;

    #[derive(Debug, PartialEq)]
    pub(crate) enum TestOperation {
        Backend(BackendEvent<Expr>),
        Diagnostics(DiagnosticsEvent<()>),
        Session(SessionEvent<()>),
    }

    type Expr = crate::model::Expr<Ident<String>, ()>;

    impl<'a> From<BackendEvent<Expr>> for TestOperation {
        fn from(event: BackendEvent<Expr>) -> Self {
            TestOperation::Backend(event)
        }
    }

    impl<'a> From<DiagnosticsEvent<()>> for TestOperation {
        fn from(event: DiagnosticsEvent<()>) -> Self {
            TestOperation::Diagnostics(event)
        }
    }

    impl<'a> From<SessionEvent<()>> for TestOperation {
        fn from(event: SessionEvent<()>) -> Self {
            TestOperation::Session(event)
        }
    }

    #[test]
    fn emit_ld_b_deref_hl() {
        use crate::model::*;
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
                BackendEvent::EmitItem(Item::Instruction(Instruction::Ld(Ld::Simple(
                    SimpleOperand::B,
                    SimpleOperand::DerefHl
                ))))
                .into()
            ]
        )
    }

    #[test]
    fn emit_rst_1_minus_1() {
        test_rst_1_op_1(BinOp::Minus)
    }

    #[test]
    fn emit_rst_1_plus_1() {
        test_rst_1_op_1(BinOp::Plus)
    }

    fn test_rst_1_op_1(op: BinOp) {
        use crate::model::*;
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
                BackendEvent::EmitItem(Item::Instruction(Instruction::Rst(Expr::from_items(&[
                    1.into(),
                    1.into(),
                    op.into()
                ]))))
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
            [
                BackendEvent::EmitItem(Item::Data(Atom::Name(label.into()).into(), Width::Word))
                    .into()
            ]
        );
    }

    #[test]
    fn analyze_label() {
        let label = "label";
        let actions =
            collect_semantic_actions(|actions| actions.enter_stmt(Some((label.into(), ()))).exit());
        assert_eq!(
            actions,
            [SessionEvent::DefineSymbol((label.into(), ()), Atom::LocationCounter.into()).into()]
        )
    }

    #[test]
    fn analyze_org_dot() {
        let actions = collect_semantic_actions(|actions| {
            let mut actions = actions
                .enter_stmt(None)
                .enter_command((Directive::Org.into(), ()))
                .add_argument();
            actions.push_atom((ExprAtom::LocationCounter, ()));
            actions.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [BackendEvent::SetOrigin(Atom::LocationCounter.into()).into()]
        );
    }

    #[test]
    fn define_named_expr() {
        let name: Ident<_> = "my_expr".into();
        let ident: Ident<_> = "id".into();
        let actions = collect_semantic_actions(|actions| {
            let mut actions = actions
                .enter_stmt(Some((name.clone(), ())))
                .enter_expr_def(())
                .next();
            actions.push_atom((ExprAtom::Ident(ident.clone()), ()));
            actions.exit().exit()
        });
        assert_eq!(
            actions,
            [SessionEvent::DefineExpr(name, vec![], Atom::Name(ident).into()).into()]
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
            TokenSeqContext::exit(params.next())
        });
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiagnostic(Message::MacroRequiresName.at(()).into()).into()]
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
            let mut token_seq_actions = params_actions.next();
            for token in body.borrow().iter().cloned().map(|t| (t, ())) {
                token_seq_actions.push_token(token)
            }
            TokenSeqContext::exit(token_seq_actions)
        });
        assert_eq!(
            actions,
            [SessionEvent::DefineMacro(
                name.into(),
                params.borrow().iter().cloned().map(Into::into).collect(),
                body.borrow().iter().cloned().collect()
            )
            .into()]
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
            [SessionEvent::InvokeMacro(name.into(), Vec::new()).into()]
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
            [SessionEvent::InvokeMacro(name.into(), vec![vec![arg_token]]).into()]
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
            [DiagnosticsEvent::EmitDiagnostic(
                Message::OperandCount {
                    actual: 1,
                    expected: 0
                }
                .at(())
                .into()
            )
            .into()]
        )
    }

    #[test]
    fn diagnose_parsing_error() {
        let diagnostic = Message::UnexpectedToken { token: () }.at(());
        let actions = collect_semantic_actions(|actions| {
            let mut stmt = actions.enter_stmt(None);
            stmt.diagnostics().emit_diagnostic(diagnostic.clone());
            stmt.exit()
        });
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiagnostic(diagnostic.into()).into()]
        )
    }

    #[test]
    fn recover_from_malformed_expr() {
        let diagnostic = Message::UnexpectedToken { token: () }.at(());
        let actions = collect_semantic_actions(|file| {
            let mut expr = file
                .enter_stmt(None)
                .enter_command((Command::Mnemonic(Mnemonic::Add), ()))
                .add_argument();
            expr.diagnostics().emit_diagnostic(diagnostic.clone());
            expr.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiagnostic(diagnostic.into()).into()]
        )
    }

    pub(super) type MockSession<'a> = crate::analysis::session::MockSession<'a, TestOperation, ()>;

    pub(crate) fn collect_semantic_actions<F>(f: F) -> Vec<TestOperation>
    where
        F: for<'a> FnOnce(TestSemanticActions<'a>) -> TestSemanticActions<'a>,
    {
        let operations = RefCell::new(Vec::new());
        let session = MockSession::new(&operations);
        f(SemanticActions::new(session));
        operations.into_inner()
    }

    type TestSemanticActions<'a> = SemanticActions<MockSession<'a>>;
}
