use super::*;

use crate::analyze::reentrancy::ReentrancyActions;
use crate::analyze::semantics::actions::TokenStreamState;
use crate::analyze::semantics::arg::*;
use crate::analyze::semantics::keywords::{Directive, Mnemonic};
use crate::analyze::semantics::resolve::NameTable;
use crate::analyze::semantics::RelocLookup;
use crate::analyze::syntax::actions::{BuiltinInstrContext, InstrFinalizer};
use crate::object::builder::Item;

use std::ops::DerefMut;

mod arg;
mod cpu_instr;
mod directive;

impl<R, V> From<BuiltinInstrState<R, V>>
    for TokenStreamState<
        <R as IdentSource>::Ident,
        <R as StringSource>::StringRef,
        <R as SpanSource>::Span,
    >
where
    R: ReentrancyActions,
{
    fn from(_: BuiltinInstrState<R, V>) -> Self {
        InstrLineState::new().into()
    }
}

impl<'a, R, N, B> BuiltinInstrContext for BuiltinInstrSemantics<'a, R, N, B>
where
    R: ReentrancyActions,
    R::Ident: 'static,
    R::StringRef: 'static,
    R::Span: 'static,
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
    type ArgContext = ArgSemantics<'a, R, N, B::ExprBuilder>;

    fn will_parse_arg(self) -> Self::ArgContext {
        self.map_builder(|builder| builder.build_const())
            .map_state(ExprBuilder::new)
    }
}

impl<'a, R, N, B> InstrFinalizer for BuiltinInstrSemantics<'a, R, N, B>
where
    R: ReentrancyActions,
    R::Ident: 'static,
    R::StringRef: 'static,
    R::Span: 'static,
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
    type Next = TokenStreamSemantics<'a, R, N, B>;

    fn did_parse_instr(self) -> Self::Next {
        let args = self.core.state.args;
        let session = set_state!(self, InstrLineState::new().into());
        match self.core.state.mnemonic.item {
            BuiltinMnemonic::CpuInstr(cpu_instr) => {
                analyze_mnemonic((&cpu_instr, self.core.state.mnemonic.span), args, session)
                    .map_state(Into::into)
            }
            BuiltinMnemonic::Directive(directive) => directive::analyze_directive(
                (directive, self.core.state.mnemonic.span),
                self.core.state.label,
                args,
                session,
            ),
        }
    }
}

impl<'a, R, N, B, S> Semantics<'a, R, N, B, S>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: NameTable<R::Ident, MacroId = R::MacroId, SymbolId = B::SymbolId>,
    B: Backend<R::Span>,
{
    pub(in crate::analyze::semantics) fn expect_const(
        &mut self,
        arg: Arg<B::Value, R::StringRef, R::Span>,
    ) -> Result<B::Value, ()> {
        match arg {
            Arg::Bare(DerefableArg::Const(value)) => Ok(value),
            Arg::Bare(DerefableArg::Symbol(_, span)) => {
                let keyword = self.strip_span(&span);
                self.emit_diag(Message::KeywordInExpr { keyword }.at(span));
                Err(())
            }
            Arg::Error => Err(()),
            _ => unimplemented!(),
        }
    }

    pub(in crate::analyze::semantics) fn define_symbol_with_params(
        &mut self,
        (name, span): (R::Ident, R::Span),
        expr: Arg<B::Value, R::StringRef, R::Span>,
    ) {
        if let Ok(value) = self.expect_const(expr) {
            let id = self.reloc_lookup(name, span.clone());
            self.core.builder.define_symbol(id, span, value);
        }
    }
}

impl From<Directive> for BuiltinMnemonic {
    fn from(directive: Directive) -> Self {
        BuiltinMnemonic::Directive(directive)
    }
}

impl From<Mnemonic> for BuiltinMnemonic {
    fn from(mnemonic: Mnemonic) -> Self {
        BuiltinMnemonic::CpuInstr(mnemonic)
    }
}

fn analyze_mnemonic<'a, R: ReentrancyActions, N, B>(
    name: (&Mnemonic, R::Span),
    args: BuiltinInstrArgs<B::Value, R::StringRef, R::Span>,
    mut session: TokenStreamSemantics<'a, R, N, B>,
) -> TokenStreamSemantics<'a, R, N, B>
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
        let operand = cpu_instr::operand::analyze_operand(arg, name.0.context(), &mut session);
        operands.push(operand)
    }
    if let Ok(instruction) = cpu_instr::analyze_instruction(name, operands, &mut session) {
        session.core.builder.emit_item(Item::CpuInstr(instruction))
    }
    session
}

#[cfg(test)]
mod tests {
    use crate::analyze::semantics::actions::tests::collect_semantic_actions;
    use crate::analyze::syntax::actions::*;
    use crate::analyze::syntax::actions::{ExprAtom::*, Operator::*};
    use crate::analyze::Literal::*;
    use crate::diag::{DiagnosticsEvent, Message, MockSpan};

    #[ignore]
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
