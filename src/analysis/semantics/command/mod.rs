use self::args::*;

use super::{Ident, Label, Literal, Params, ParamsAdapter, SemanticActions, StmtActions};

use crate::analysis::backend::{Finish, FinishFnDef, LocationCounter, Name, PushOp};
use crate::analysis::session::Session;
use crate::analysis::syntax::*;
use crate::diag::span::{MergeSpans, StripSpan};
use crate::diag::{CompactDiag, Diagnostics, EmitDiag, Message};
use crate::model::{BinOp, FnCall, Item};

mod args;
mod directive;
mod mnemonic;
mod operand;

pub(in crate::analysis) struct CommandActions<S: Session> {
    parent: StmtActions<S>,
    command: (Command, S::Span),
    args: CommandArgs<S::StringRef, S::Span>,
    has_errors: bool,
}

impl<S: Session> CommandActions<S> {
    pub(super) fn new(parent: StmtActions<S>, command: (Command, S::Span)) -> CommandActions<S> {
        CommandActions {
            parent,
            command,
            args: Vec::new(),
            has_errors: false,
        }
    }
}

impl<S: Session> MergeSpans<S::Span> for CommandActions<S> {
    fn merge_spans(&mut self, left: &S::Span, right: &S::Span) -> S::Span {
        self.parent.merge_spans(left, right)
    }
}

impl<S: Session> StripSpan<S::Span> for CommandActions<S> {
    type Stripped = S::Stripped;

    fn strip_span(&mut self, span: &S::Span) -> Self::Stripped {
        self.parent.strip_span(span)
    }
}

impl<S: Session> EmitDiag<S::Span, S::Stripped> for CommandActions<S> {
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<S::Span, S::Stripped>>) {
        self.has_errors = true;
        self.parent.emit_diag(diag)
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
            let prepared = PreparedCommand::new(self.command, &mut self.parent);
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

delegate_diagnostics! {
    {R, S, P: Diagnostics<S>}, ExprBuilder<R, S, P>, {parent}, P, S
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
    Self: Diagnostics<S>,
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

    fn apply_operator(&mut self, (op, span): (Operator, S)) {
        let variant = match op {
            Operator::Unary(UnaryOperator::Parentheses) => {
                let inner = self.pop();
                ArgVariant::Unary(ArgUnaryOp::Parentheses, Box::new(inner))
            }
            Operator::Binary(binary) => {
                let rhs = self.pop();
                let lhs = self.pop();
                ArgVariant::Binary(binary, Box::new(lhs), Box::new(rhs))
            }
            Operator::FnCall(n) => {
                let args = self.stack.split_off(self.stack.len() - n);
                let name = self.pop();
                let name = (
                    match name.variant {
                        ArgVariant::Atom(ArgAtom::Ident(ident)) => ident,
                        _ => {
                            self.emit_diag(Message::OnlyIdentsCanBeCalled.at(name.span));
                            return;
                        }
                    },
                    name.span,
                );
                ArgVariant::FnCall(name, args)
            }
        };
        self.stack.push(Arg { variant, span })
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
            actions.build_value(&Default::default(), |builder| {
                operand::analyze_operand(arg, name.0.context(), builder)
            })
        })
        .collect();
    if let Ok(instruction) = mnemonic::analyze_instruction(name, operands, actions) {
        actions.session().emit_item(Item::Instruction(instruction))
    }
}

impl<S: Session> SemanticActions<S> {
    fn analyze_expr(
        &mut self,
        params: &Params<S::StringRef, S::Span>,
        expr: Arg<S::StringRef, S::Span>,
    ) -> Result<S::Value, ()> {
        self.build_value(params, |mut builder| {
            let result = builder.eval_arg(expr);
            let (session, value) = builder.finish();
            (session, result.map(|()| value))
        })
    }

    fn define_symbol(
        &mut self,
        (name, span): (Ident<S::StringRef>, S::Span),
        params: &Params<S::StringRef, S::Span>,
        expr: Arg<S::StringRef, S::Span>,
    ) -> Result<(), ()> {
        let mut result = Ok(());
        self.with_session(|session| {
            let builder = session.define_symbol(name, span);
            let mut adapter = ParamsAdapter::new(builder, params);
            result = adapter.eval_arg(expr);
            adapter.finish_fn_def()
        });
        result
    }
}

trait EvalArg<I, S: Clone> {
    fn eval_arg(&mut self, arg: Arg<I, S>) -> Result<(), ()>;
}

trait ArgEvaluator<N, S: Clone>:
    PushOp<LocationCounter, S>
    + PushOp<i32, S>
    + PushOp<Name<N>, S>
    + PushOp<BinOp, S>
    + PushOp<FnCall, S>
    + Diagnostics<S>
{
}

impl<T, N, S: Clone> ArgEvaluator<N, S> for T where
    Self: PushOp<LocationCounter, S>
        + PushOp<i32, S>
        + PushOp<Name<N>, S>
        + PushOp<BinOp, S>
        + PushOp<FnCall, S>
        + Diagnostics<S>
{
}

impl<'a, T, R, S> EvalArg<R, S> for T
where
    T: ArgEvaluator<Ident<R>, S>,
    R: Eq,
    S: Clone,
{
    fn eval_arg(&mut self, arg: Arg<R, S>) -> Result<(), ()> {
        match arg.variant {
            ArgVariant::Atom(ArgAtom::Ident(ident)) => {
                self.push_op(Name(ident), arg.span);
                Ok(())
            }
            ArgVariant::Atom(ArgAtom::Literal(Literal::Number(n))) => {
                self.push_op(n, arg.span);
                Ok(())
            }
            ArgVariant::Atom(ArgAtom::Literal(Literal::Operand(_))) => {
                Err(Message::KeywordInExpr {
                    keyword: self.strip_span(&arg.span),
                }
                .at(arg.span))
            }
            ArgVariant::Atom(ArgAtom::Literal(Literal::String(_))) => {
                Err(Message::StringInInstruction.at(arg.span))
            }
            ArgVariant::Atom(ArgAtom::LocationCounter) => {
                self.push_op(LocationCounter, arg.span);
                Ok(())
            }
            ArgVariant::Unary(ArgUnaryOp::Parentheses, expr) => Ok(self.eval_arg(*expr)?),
            ArgVariant::Binary(binary, left, right) => {
                self.eval_arg(*left)?;
                self.eval_arg(*right)?;
                self.push_op(binary, arg.span);
                Ok(())
            }
            ArgVariant::FnCall((name, span), args) => {
                let n = args.len();
                for arg in args {
                    self.eval_arg(arg)?;
                }
                self.push_op(Name(name), span.clone());
                self.push_op(FnCall(n), span);
                Ok(())
            }
        }
        .map_err(|diagnostic| {
            self.emit_diag(diagnostic);
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::analysis::semantics::tests::collect_semantic_actions;
    use crate::analysis::syntax::*;
    use crate::analysis::syntax::{Command::*, Directive::*, ExprAtom::*, Operator::*};
    use crate::analysis::Literal::*;
    use crate::diag::{DiagnosticsEvent, Message, MockSpan};

    #[test]
    fn diagnose_literal_as_fn_name() {
        assert_eq!(
            collect_semantic_actions::<_, MockSpan<_>>(|actions| {
                let mut actions = actions
                    .enter_unlabeled_stmt()
                    .enter_command((Directive(Db), "db".into()))
                    .add_argument();
                actions.push_atom((Literal(Number(7)), "literal".into()));
                actions.apply_operator((FnCall(0), "call".into()));
                actions.exit().exit().exit()
            }),
            [DiagnosticsEvent::EmitDiag(
                Message::OnlyIdentsCanBeCalled.at("literal".into()).into()
            )
            .into()]
        );
    }
}
