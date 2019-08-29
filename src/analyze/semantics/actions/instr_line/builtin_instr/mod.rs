use super::*;

use crate::analyze::reentrancy::ReentrancyActions;
use crate::analyze::semantics::actions::TokenStreamState;
use crate::analyze::semantics::arg::*;
use crate::analyze::semantics::builtin_instr::cpu_instr::analyze_instruction;
use crate::analyze::semantics::builtin_instr::cpu_instr::operand::analyze_operand;
use crate::analyze::semantics::builtin_instr::directive::analyze_directive;
use crate::analyze::semantics::builtin_instr::directive::*;
use crate::analyze::semantics::builtin_instr::BuiltinInstrMnemonic;
use crate::analyze::semantics::resolve::NameTable;
use crate::analyze::semantics::{Params, RelocLookup, ResolveNames, WithParams};
use crate::analyze::syntax::actions::{BuiltinInstrActions, InstrFinalizer};
use crate::object::builder::{Finish, Item};

use std::ops::DerefMut;

mod arg;

impl<R: ReentrancyActions> From<BuiltinInstrState<R>> for TokenStreamState<R> {
    fn from(state: BuiltinInstrState<R>) -> Self {
        state.parent.into()
    }
}

impl<R, N, B> BuiltinInstrActions<R::Ident, Literal<R::StringRef>, R::Span>
    for BuiltinInstrSemantics<R, N, B>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: Backend<R::Span>,
{
    type ArgActions = ArgSemantics<R, N, B>;

    fn will_parse_arg(self) -> Self::ArgActions {
        self.map_state(ExprBuilder::new)
    }
}

impl<R, N, B> InstrFinalizer<R::Span> for BuiltinInstrSemantics<R, N, B>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: Backend<R::Span>,
{
    type Next = TokenStreamSemantics<R, N, B>;

    fn did_parse_instr(self) -> Self::Next {
        let args = self.state.args;
        let mut semantics = set_state!(self, self.state.parent);
        let instr = BuiltinInstr::new(self.state.mnemonic, &mut semantics);
        semantics = semantics.flush_label();
        instr.exec(args, semantics)
    }
}

enum BuiltinInstr<S: ReentrancyActions> {
    Binding(
        (BindingDirective, S::Span),
        Option<Label<S::Ident, S::Span>>,
    ),
    Directive((SimpleDirective, S::Span)),
    CpuInstr((Mnemonic, S::Span)),
}

impl<R: ReentrancyActions> BuiltinInstr<R> {
    fn new<N, B>(
        (mnemonic, span): (BuiltinInstrMnemonic, R::Span),
        stmt: &mut InstrLineSemantics<R, N, B>,
    ) -> Self {
        match mnemonic {
            BuiltinInstrMnemonic::Directive(Directive::Binding(binding)) => {
                BuiltinInstr::Binding((binding, span), stmt.state.label.take())
            }
            BuiltinInstrMnemonic::Directive(Directive::Simple(simple)) => {
                BuiltinInstr::Directive((simple, span))
            }
            BuiltinInstrMnemonic::CpuInstr(cpu_instr) => BuiltinInstr::CpuInstr((cpu_instr, span)),
        }
    }

    fn exec<N, B>(
        self,
        args: BuiltinInstrArgs<R::Ident, R::StringRef, R::Span>,
        session: InstrLineSemantics<R, N, B>,
    ) -> TokenStreamSemantics<R, N, B>
    where
        N: DerefMut,
        N::Target: StartScope<R::Ident>
            + NameTable<
                R::Ident,
                Keyword = &'static Keyword,
                MacroId = R::MacroId,
                SymbolId = B::SymbolId,
            >,
        B: Backend<R::Span>,
    {
        match self {
            BuiltinInstr::Binding((binding, span), label) => {
                analyze_directive((Directive::Binding(binding), span), label, args, session)
            }
            BuiltinInstr::Directive((simple, span)) => {
                analyze_directive((Directive::Simple(simple), span), None, args, session)
            }
            BuiltinInstr::CpuInstr(mnemonic) => {
                analyze_mnemonic(mnemonic, args, session).map_state(Into::into)
            }
        }
    }
}

fn analyze_mnemonic<R: ReentrancyActions, N, B>(
    name: (Mnemonic, R::Span),
    args: BuiltinInstrArgs<R::Ident, R::StringRef, R::Span>,
    mut session: InstrLineSemantics<R, N, B>,
) -> InstrLineSemantics<R, N, B>
where
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: Backend<R::Span>,
{
    let mut operands = Vec::new();
    for arg in args {
        let builder = session.map_builder(Backend::build_const).resolve_names();
        let (operand, returned_session) = analyze_operand(arg, name.0.context(), builder);
        session = returned_session;
        operands.push(operand)
    }
    if let Ok(instruction) = analyze_instruction(name, operands, &mut session) {
        session.builder.emit_item(Item::CpuInstr(instruction))
    }
    session
}

impl<R, N, B, S> Session<R, N, B, S>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: NameTable<R::Ident, MacroId = R::MacroId, SymbolId = B::SymbolId>,
    B: Backend<R::Span>,
{
    pub(in crate::analyze::semantics) fn analyze_expr(
        self,
        expr: Arg<R::Ident, R::StringRef, R::Span>,
    ) -> (Result<B::Value, ()>, Self) {
        let mut builder = self.map_builder(Backend::build_const).resolve_names();
        let result = builder.eval_arg(expr);
        let (session, value) = builder.finish();
        (result.map(|()| value), session)
    }

    pub(in crate::analyze::semantics) fn define_symbol_with_params(
        mut self,
        (name, span): (R::Ident, R::Span),
        params: &Params<R::Ident, R::Span>,
        expr: Arg<R::Ident, R::StringRef, R::Span>,
    ) -> (Result<(), ()>, Self) {
        let id = self.reloc_lookup(name, span.clone());
        let mut builder = self
            .map_builder(|builder| builder.define_symbol(id, span))
            .resolve_names()
            .with_params(params);
        let result = builder.eval_arg(expr);
        let (session, ()) = builder.finish();
        (result, session)
    }
}

#[cfg(test)]
mod tests {
    use crate::analyze::semantics::actions::tests::collect_semantic_actions;
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
