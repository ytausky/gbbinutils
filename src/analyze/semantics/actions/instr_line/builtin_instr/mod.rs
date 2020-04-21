use super::*;

use crate::analyze::reentrancy::ReentrancyActions;
use crate::analyze::semantics::actions::TokenStreamState;
use crate::analyze::semantics::arg::*;
use crate::analyze::semantics::builtin_instr::DispatchBuiltinInstrLine;
use crate::analyze::semantics::resolve::NameTable;
use crate::analyze::semantics::RelocLookup;
use crate::analyze::syntax::actions::{BuiltinInstrActions, InstrFinalizer};

use std::ops::DerefMut;

mod arg;

impl<I, R, V> From<BuiltinInstrState<I, R, V>>
    for TokenStreamState<
        <R as IdentSource>::Ident,
        <R as StringSource>::StringRef,
        <R as SpanSource>::Span,
    >
where
    I: BuiltinInstrSet<R>,
    R: ReentrancyActions,
{
    fn from(_: BuiltinInstrState<I, R, V>) -> Self {
        InstrLineState::new().into()
    }
}

impl<I, R, N, B> BuiltinInstrActions<R::Ident, Literal<R::StringRef>, R::Span>
    for BuiltinInstrSemantics<I, R, N, B>
where
    I: BuiltinInstrSet<R>,
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword<I::Binding, I::Free>,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: Backend<R::Span>,
    Self: DispatchBuiltinInstrLine<I, R, N, B>,
{
    type ArgActions = ArgSemantics<I, R, N, B::ExprBuilder>;

    fn will_parse_arg(self) -> Self::ArgActions {
        self.map_builder(|builder| builder.build_const())
            .map_state(ExprBuilder::new)
    }
}

impl<I, R, N, B> InstrFinalizer<R::Span> for BuiltinInstrSemantics<I, R, N, B>
where
    I: BuiltinInstrSet<R>,
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword<I::Binding, I::Free>,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: Backend<R::Span>,
    Self: DispatchBuiltinInstrLine<I, R, N, B>,
{
    type Next = TokenStreamSemantics<I, R, N, B>;

    fn did_parse_instr(self) -> Self::Next {
        self.dispatch_builtin_instr_line()
    }
}

impl<I, R, N, B, S> Session<I, R, N, B, S>
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
            self.builder.define_symbol(id, span, value);
        }
    }
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
