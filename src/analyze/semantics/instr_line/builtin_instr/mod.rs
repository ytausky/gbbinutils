pub(super) use self::directive::{BindingDirective, Directive, SimpleDirective};
pub(in crate::analyze) use self::operand::OperandSymbol;

use self::arg::*;

use super::*;

use crate::analyze::semantics::{Keyword, Params, RelocLookup, ResolveNames, WithParams};
use crate::analyze::session::ReentrancyActions;
use crate::analyze::syntax::actions::{BuiltinInstrActions, InstrFinalizer};
use crate::diag::{Diagnostics, EmitDiag, Message};
use crate::expr::{BinOp, FnCall, LocationCounter};
use crate::object::builder::{Finish, Item, Name, PushOp};

pub(in crate::analyze::semantics) mod cpu_instr;
pub(in crate::analyze::semantics) mod directive;
pub(in crate::analyze::semantics) mod operand;

mod arg;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum BuiltinInstr {
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

pub(in crate::analyze) struct BuiltinInstrState<S: ReentrancyActions> {
    parent: InstrLineState<S>,
    command: (BuiltinInstr, S::Span),
    args: BuiltinInstrArgs<S::Ident, S::StringRef, S::Span>,
}

impl<S: ReentrancyActions> BuiltinInstrState<S> {
    pub(super) fn new(parent: InstrLineState<S>, command: (BuiltinInstr, S::Span)) -> Self {
        Self {
            parent,
            command,
            args: Vec::new(),
        }
    }
}

impl<S> BuiltinInstrActions<S::Ident, Literal<S::StringRef>, S::Span> for BuiltinInstrSemantics<S>
where
    S: ReentrancyActions<Keyword = &'static Keyword>,
{
    type ArgActions = ArgSemantics<S>;

    fn will_parse_arg(self) -> Self::ArgActions {
        self.map_line(ExprBuilder::new)
    }
}

impl<S: ReentrancyActions<Keyword = &'static Keyword>> InstrFinalizer<S::Span>
    for BuiltinInstrSemantics<S>
{
    type Next = TokenStreamSemantics<S>;

    fn did_parse_instr(self) -> Self::Next {
        let args = self.state.args;
        let mut semantics = set_state!(self, self.state.parent);
        let prepared = PreparedBuiltinInstr::new(self.state.command, &mut semantics);
        semantics = semantics.flush_label();
        prepared.exec(args, semantics.session)
    }
}

enum PreparedBuiltinInstr<S: ReentrancyActions> {
    Binding(
        (BindingDirective, S::Span),
        Option<Label<S::Ident, S::Span>>,
    ),
    Directive((SimpleDirective, S::Span)),
    Mnemonic((Mnemonic, S::Span)),
}

impl<S: ReentrancyActions<Keyword = &'static Keyword>> PreparedBuiltinInstr<S> {
    fn new((command, span): (BuiltinInstr, S::Span), stmt: &mut InstrLineSemantics<S>) -> Self {
        match command {
            BuiltinInstr::Directive(Directive::Binding(binding)) => {
                PreparedBuiltinInstr::Binding((binding, span), stmt.state.label.take())
            }
            BuiltinInstr::Directive(Directive::Simple(simple)) => {
                PreparedBuiltinInstr::Directive((simple, span))
            }
            BuiltinInstr::Mnemonic(mnemonic) => PreparedBuiltinInstr::Mnemonic((mnemonic, span)),
        }
    }

    fn exec(
        self,
        args: BuiltinInstrArgs<S::Ident, S::StringRef, S::Span>,
        session: S,
    ) -> TokenStreamSemantics<S> {
        match self {
            PreparedBuiltinInstr::Binding((binding, span), label) => directive::analyze_directive(
                (Directive::Binding(binding), span),
                label,
                args,
                session,
            ),
            PreparedBuiltinInstr::Directive((simple, span)) => {
                directive::analyze_directive((Directive::Simple(simple), span), None, args, session)
            }
            PreparedBuiltinInstr::Mnemonic(mnemonic) => {
                TokenStreamSemantics::new(analyze_mnemonic(mnemonic, args, session))
            }
        }
    }
}

fn analyze_mnemonic<S: ReentrancyActions>(
    name: (Mnemonic, S::Span),
    args: BuiltinInstrArgs<S::Ident, S::StringRef, S::Span>,
    mut session: S,
) -> S {
    let mut operands = Vec::new();
    for arg in args {
        let builder = session.build_const().resolve_names();
        let (operand, returned_session) = operand::analyze_operand(arg, name.0.context(), builder);
        session = returned_session;
        operands.push(operand)
    }
    if let Ok(instruction) = cpu_instr::analyze_instruction(name, operands, &mut session) {
        session.emit_item(Item::CpuInstr(instruction))
    }
    session
}

trait SessionExt: ReentrancyActions {
    fn analyze_expr(
        self,
        expr: Arg<Self::Ident, Self::StringRef, Self::Span>,
    ) -> (Result<Self::Value, ()>, Self) {
        let mut builder = self.build_const().resolve_names();
        let result = builder.eval_arg(expr);
        let (session, value) = builder.finish();
        (result.map(|()| value), session)
    }

    fn define_symbol_with_params(
        mut self,
        (name, span): (Self::Ident, Self::Span),
        params: &Params<Self::Ident, Self::Span>,
        expr: Arg<Self::Ident, Self::StringRef, Self::Span>,
    ) -> (Result<(), ()>, Self) {
        let id = self.reloc_lookup(name, span.clone());
        let mut builder = self
            .define_symbol(id, span)
            .resolve_names()
            .with_params(params);
        let result = builder.eval_arg(expr);
        let (session, ()) = builder.finish();
        (result, session)
    }
}

impl<S: ReentrancyActions> SessionExt for S {}

trait EvalArg<I, R, S: Clone> {
    fn eval_arg(&mut self, arg: Arg<I, R, S>) -> Result<(), ()>;
}

impl<'a, T, I, R, S> EvalArg<I, R, S> for T
where
    T: PushOp<LocationCounter, S>
        + PushOp<i32, S>
        + PushOp<Name<I>, S>
        + PushOp<BinOp, S>
        + PushOp<FnCall, S>
        + Diagnostics<S>,
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
    use crate::analyze::semantics::tests::collect_semantic_actions;
    use crate::analyze::syntax::actions::*;
    use crate::analyze::syntax::actions::{ExprAtom::*, Operator::*};
    use crate::analyze::Literal::*;
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
