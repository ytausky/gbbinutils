use self::args::*;

use super::{Ident, Label, Literal, SemanticActions, StmtActions};

use crate::analysis::backend::{LocationCounter, ValueBuilder};
use crate::analysis::session::{Finish, Session};
use crate::diag::span::{MergeSpans, StripSpan};
use crate::diag::{CompactDiagnostic, DelegateDiagnostics, EmitDiagnostic, Message};
use crate::model::Item;
use crate::syntax::*;

mod args;
mod directive;
mod mnemonic;
mod operand;

pub(crate) struct CommandActions<S: Session> {
    parent: StmtActions<S>,
    name: (Command, S::Span),
    args: CommandArgs<S::StringRef, S::Span>,
    has_errors: bool,
}

impl<S: Session> CommandActions<S> {
    pub(super) fn new(parent: StmtActions<S>, name: (Command, S::Span)) -> CommandActions<S> {
        CommandActions {
            parent,
            name,
            args: Vec::new(),
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

impl<S: Session> CommandContext<S::Span> for CommandActions<S> {
    type Ident = Ident<S::StringRef>;
    type Command = Command;
    type Literal = Literal<S::StringRef>;
    type ArgContext = ExprBuilder<S::StringRef, S::Span, Self>;
    type Parent = StmtActions<S>;

    fn add_argument(self) -> Self::ArgContext {
        ExprBuilder {
            stack: Vec::new(),
            parent: self,
        }
    }

    fn exit(mut self) -> Self::Parent {
        if !self.has_errors {
            let prepared = PreparedCommand::new(self.name, &mut self.parent);
            self.parent.define_label_if_present();
            prepared.exec(self.args, &mut self.parent.parent)
        }
        self.parent
    }
}

enum PreparedCommand<S: Session> {
    Binding((Directive, S::Span), Option<Label<S::StringRef, S::Span>>),
    Directive((Directive, S::Span)),
    Mnemonic((Mnemonic, S::Span)),
}

impl<S: Session> PreparedCommand<S> {
    fn new((command, span): (Command, S::Span), stmt: &mut StmtActions<S>) -> Self {
        match command {
            Command::Directive(directive) if directive.requires_symbol() => {
                PreparedCommand::Binding((directive, span), stmt.label.take())
            }
            Command::Directive(directive) => PreparedCommand::Directive((directive, span)),
            Command::Mnemonic(mnemonic) => PreparedCommand::Mnemonic((mnemonic, span)),
        }
    }

    fn exec(self, args: CommandArgs<S::StringRef, S::Span>, actions: &mut SemanticActions<S>) {
        match self {
            PreparedCommand::Binding(binding, label) => {
                directive::analyze_directive(binding, label, args, actions)
            }
            PreparedCommand::Directive(directive) => {
                directive::analyze_directive(directive, None, args, actions)
            }
            PreparedCommand::Mnemonic(mnemonic) => analyze_mnemonic(mnemonic, args, actions),
        }
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

pub(crate) struct ExprBuilder<R, S, P> {
    stack: Vec<Arg<R, S>>,
    parent: P,
}

impl<R, S, P> ExprBuilder<R, S, P> {
    fn pop(&mut self) -> Arg<R, S> {
        self.stack.pop().unwrap_or_else(|| unreachable!())
    }
}

impl<R, S, P> DelegateDiagnostics<S> for ExprBuilder<R, S, P>
where
    P: DelegateDiagnostics<S>,
{
    type Delegate = P::Delegate;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.diagnostics()
    }
}

impl<S: Session> FinalContext for ExprBuilder<S::StringRef, S::Span, CommandActions<S>> {
    type ReturnTo = CommandActions<S>;

    fn exit(mut self) -> Self::ReturnTo {
        if !self.parent.has_errors {
            assert_eq!(self.stack.len(), 1);
            self.parent.args.push(self.stack.pop().unwrap());
        }
        self.parent
    }
}

impl<R, S, P> ExprContext<S> for ExprBuilder<R, S, P>
where
    S: Clone,
    Self: DelegateDiagnostics<S>,
{
    type Ident = Ident<R>;
    type Literal = Literal<R>;

    fn push_atom(&mut self, atom: (ExprAtom<Self::Ident, Self::Literal>, S)) {
        self.stack.push(Arg {
            variant: ArgVariant::Atom(match atom.0 {
                ExprAtom::Ident(ident) => ArgAtom::Ident(ident),
                ExprAtom::Literal(literal) => ArgAtom::Literal(literal),
                ExprAtom::LocationCounter => ArgAtom::LocationCounter,
            }),
            span: atom.1,
        })
    }

    fn apply_operator(&mut self, operator: (Operator, S)) {
        match operator.0 {
            Operator::Unary(UnaryOperator::Parentheses) => {
                let inner = self.pop();
                self.stack.push(Arg {
                    variant: ArgVariant::Unary(ArgUnaryOp::Parentheses, Box::new(inner)),
                    span: operator.1,
                })
            }
            Operator::Binary(binary) => {
                let rhs = self.pop();
                let lhs = self.pop();
                self.stack.push(Arg {
                    variant: ArgVariant::Binary(binary, Box::new(lhs), Box::new(rhs)),
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
    if let Ok(instruction) = mnemonic::analyze_instruction(name, operands, actions.diagnostics()) {
        actions.session().emit_item(Item::Instruction(instruction))
    }
}

impl<S: Session> SemanticActions<S> {
    fn analyze_expr(&mut self, expr: Arg<S::StringRef, S::Span>) -> Result<S::Value, ()> {
        self.build_value(|mut builder| {
            let result = builder.analyze_expr(expr);
            let (session, value) = builder.finish();
            (session, result.map(|()| value))
        })
    }
}

trait AnalyzeExpr<I, S: Clone> {
    fn analyze_expr(&mut self, expr: Arg<I, S>) -> Result<(), ()>;
}

impl<'a, T, I, S> AnalyzeExpr<I, S> for T
where
    T: ValueBuilder<Ident<I>, S> + DelegateDiagnostics<S>,
    S: Clone,
{
    fn analyze_expr(&mut self, expr: Arg<I, S>) -> Result<(), ()> {
        match expr.variant {
            ArgVariant::Atom(ArgAtom::Ident(ident)) => {
                self.push_op(ident, expr.span);
                Ok(())
            }
            ArgVariant::Atom(ArgAtom::Literal(Literal::Number(n))) => {
                self.push_op(n, expr.span);
                Ok(())
            }
            ArgVariant::Atom(ArgAtom::Literal(Literal::Operand(_))) => {
                Err(Message::KeywordInExpr {
                    keyword: self.diagnostics().strip_span(&expr.span),
                }
                .at(expr.span))
            }
            ArgVariant::Atom(ArgAtom::Literal(Literal::String(_))) => {
                Err(Message::StringInInstruction.at(expr.span))
            }
            ArgVariant::Atom(ArgAtom::LocationCounter) => {
                self.push_op(LocationCounter, expr.span);
                Ok(())
            }
            ArgVariant::Unary(ArgUnaryOp::Parentheses, expr) => Ok(self.analyze_expr(*expr)?),
            ArgVariant::Binary(binary, left, right) => {
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
