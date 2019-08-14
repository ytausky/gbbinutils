pub(super) use self::directive::Directive;

use self::arg::*;
use self::operand::OperandSymbol;

use super::*;

use crate::analysis::semantics::{Params, RelocLookup, ResolveNames, WithParams};
use crate::analysis::session::Session;
use crate::analysis::syntax::actions::{BuiltinInstrActions, InstrFinalizer};
use crate::diag::{Diagnostics, EmitDiag, Message};
use crate::model::{BinOp, FnCall, LocationCounter};
use crate::object::builder::{Finish, Item, Name, PushOp};

pub(super) mod cpu_instr;

mod arg;
mod directive;
mod operand;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analysis) enum BuiltinInstr {
    Directive(Directive),
    Mnemonic(Mnemonic),
}

impl From<Directive> for BuiltinInstr {
    fn from(directive: Directive) -> Self {
        BuiltinInstr::Directive(directive)
    }
}

impl From<Mnemonic> for BuiltinInstr {
    fn from(mnemonic: Mnemonic) -> Self {
        BuiltinInstr::Mnemonic(mnemonic)
    }
}

pub(super) type BuiltinInstrSemantics<S> = SemanticActions<BuiltinInstrState<S>, S>;

pub(in crate::analysis) struct BuiltinInstrState<S: Session> {
    parent: InstrLineState<S>,
    command: (BuiltinInstr, S::Span),
    args: BuiltinInstrArgs<S::Ident, S::StringRef, S::Span>,
}

impl<S: Session> BuiltinInstrState<S> {
    pub(super) fn new(parent: InstrLineState<S>, command: (BuiltinInstr, S::Span)) -> Self {
        Self {
            parent,
            command,
            args: Vec::new(),
        }
    }
}

impl<S: Session> BuiltinInstrActions<S::Ident, Literal<S::StringRef>, S::Span>
    for BuiltinInstrSemantics<S>
{
    type ArgActions = ArgSemantics<S>;

    fn will_parse_arg(self) -> Self::ArgActions {
        self.map_line(ExprBuilder::new)
    }
}

impl<S: Session> InstrFinalizer<S::Span> for BuiltinInstrSemantics<S> {
    type Next = TokenStreamSemantics<S>;

    fn did_parse_instr(self) -> Self::Next {
        let args = self.state.args;
        let mut semantics = set_state!(self, self.state.parent);
        let prepared = PreparedBuiltinInstr::new(self.state.command, &mut semantics);
        semantics = semantics.define_label_if_present();
        prepared.exec(args, semantics)
    }
}

enum PreparedBuiltinInstr<S: Session> {
    Binding((Directive, S::Span), Option<Label<S::Ident, S::Span>>),
    Directive((Directive, S::Span)),
    Mnemonic((Mnemonic, S::Span)),
}

impl<S: Session> PreparedBuiltinInstr<S> {
    fn new((command, span): (BuiltinInstr, S::Span), stmt: &mut InstrLineSemantics<S>) -> Self {
        match command {
            BuiltinInstr::Directive(directive) if directive.requires_symbol() => {
                PreparedBuiltinInstr::Binding((directive, span), stmt.state.label.take())
            }
            BuiltinInstr::Directive(directive) => {
                PreparedBuiltinInstr::Directive((directive, span))
            }
            BuiltinInstr::Mnemonic(mnemonic) => PreparedBuiltinInstr::Mnemonic((mnemonic, span)),
        }
    }

    fn exec(
        self,
        args: BuiltinInstrArgs<S::Ident, S::StringRef, S::Span>,
        actions: InstrLineSemantics<S>,
    ) -> TokenStreamSemantics<S> {
        match self {
            PreparedBuiltinInstr::Binding(binding, label) => {
                directive::analyze_directive(binding, label, args, actions)
            }
            PreparedBuiltinInstr::Directive(directive) => {
                directive::analyze_directive(directive, None, args, actions)
            }
            PreparedBuiltinInstr::Mnemonic(mnemonic) => {
                analyze_mnemonic(mnemonic, args, actions).map_line(Into::into)
            }
        }
    }
}

impl Directive {
    fn requires_symbol(self) -> bool {
        match self {
            Directive::Equ | Directive::Macro | Directive::Section => true,
            _ => false,
        }
    }
}

fn analyze_mnemonic<S: Session>(
    name: (Mnemonic, S::Span),
    args: BuiltinInstrArgs<S::Ident, S::StringRef, S::Span>,
    mut actions: InstrLineSemantics<S>,
) -> InstrLineSemantics<S> {
    let mut operands = Vec::new();
    for arg in args {
        let (operand, returned_actions) = actions.build_value(&Default::default(), |builder| {
            operand::analyze_operand(arg, name.0.context(), builder)
        });
        actions = returned_actions;
        operands.push(operand)
    }
    if let Ok(instruction) = cpu_instr::analyze_instruction(name, operands, &mut actions) {
        actions.session.emit_item(Item::CpuInstr(instruction))
    }
    actions
}

impl<S: Session> InstrLineSemantics<S> {
    fn analyze_expr(
        self,
        params: &Params<S::Ident, S::Span>,
        expr: Arg<S::Ident, S::StringRef, S::Span>,
    ) -> (Result<S::Value, ()>, Self) {
        let (value, actions) = self.build_value(params, |mut builder| {
            let result = builder.eval_arg(expr);
            let (session, value) = builder.finish();
            (result.map(|()| value), session)
        });
        (value, actions)
    }

    fn define_symbol(
        mut self,
        (name, span): (S::Ident, S::Span),
        params: &Params<S::Ident, S::Span>,
        expr: Arg<S::Ident, S::StringRef, S::Span>,
    ) -> (Result<(), ()>, Self) {
        let id = self.session.reloc_lookup(name, span.clone());
        let mut builder = self
            .session
            .define_symbol(id, span)
            .resolve_names()
            .with_params(params);
        let result = builder.eval_arg(expr);
        let (session, ()) = builder.finish();
        self.session = session;
        (result, self)
    }
}

trait EvalArg<I, R, S: Clone> {
    fn eval_arg(&mut self, arg: Arg<I, R, S>) -> Result<(), ()>;
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

impl<'a, T, I, R, S> EvalArg<I, R, S> for T
where
    T: ArgEvaluator<I, S>,
    R: Eq,
    S: Clone,
{
    fn eval_arg(&mut self, arg: Arg<I, R, S>) -> Result<(), ()> {
        match arg.variant {
            ArgVariant::Atom(ArgAtom::Error) => Err(())?,
            ArgVariant::Atom(ArgAtom::Ident(ident)) => {
                self.push_op(Name(ident), arg.span);
                Ok(())
            }
            ArgVariant::Atom(ArgAtom::Literal(Literal::Number(n))) => {
                self.push_op(n, arg.span);
                Ok(())
            }
            ArgVariant::Atom(ArgAtom::OperandSymbol(_)) => Err(Message::KeywordInExpr {
                keyword: self.strip_span(&arg.span),
            }
            .at(arg.span)),
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
                self.push_op(Name(name.ok_or(())?), span.clone());
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
    use crate::analysis::syntax::actions::*;
    use crate::analysis::syntax::actions::{ExprAtom::*, Operator::*};
    use crate::analysis::Literal::*;
    use crate::diag::{DiagnosticsEvent, Message, MockSpan};

    #[test]
    fn diagnose_literal_as_fn_name() {
        assert_eq!(
            collect_semantic_actions::<_, MockSpan<_>>(|actions| {
                let mut actions = actions
                    .will_parse_line()
                    .into_instr_line()
                    .will_parse_instr("DB".into(), "db".into())
                    .into_builtin_instr()
                    .will_parse_arg();
                actions.act_on_atom(Literal(Number(7)), "literal".into());
                actions.act_on_operator(FnCall(0), "call".into());
                actions
                    .did_parse_arg()
                    .did_parse_instr()
                    .did_parse_line("eol".into())
                    .act_on_eos("eos".into())
            }),
            [DiagnosticsEvent::EmitDiag(
                Message::OnlyIdentsCanBeCalled.at("literal".into()).into()
            )
            .into()]
        );
    }
}
