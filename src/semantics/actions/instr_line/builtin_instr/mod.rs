use super::*;

use crate::semantics::actions::TokenStreamState;
use crate::semantics::arg::*;
use crate::semantics::keywords::{Directive, Mnemonic};
use crate::semantics::RelocLookup;
use crate::session::builder::Item;
use crate::session::resolve::NameTable;
use crate::syntax::actions::{BuiltinInstrContext, InstrFinalizer};

mod arg;
mod cpu_instr;
mod directive;

impl<S: Session> From<BuiltinInstrState<S>>
    for TokenStreamState<
        <S as IdentSource>::Ident,
        <S as StringSource>::StringRef,
        <S as SpanSource>::Span,
    >
{
    fn from(_: BuiltinInstrState<S>) -> Self {
        InstrLineState::new().into()
    }
}

impl<'a, S: Session> BuiltinInstrContext for BuiltinInstrSemantics<'a, S>
where
    S::Ident: 'static,
    S::StringRef: 'static,
    S::Span: 'static,
    S::ExprBuilder: StartScope<S::Ident>
        + NameTable<
            S::Ident,
            Keyword = &'static Keyword,
            MacroId = S::MacroId,
            SymbolId = S::SymbolId,
        > + Diagnostics<S::Span, Stripped = S::Stripped>,
{
    type ArgContext = ArgSemantics<'a, S::ExprBuilder>;

    fn will_parse_arg(self) -> Self::ArgContext {
        Semantics {
            session: self.session.build_const(),
            state: ExprBuilder::new(self.state),
            tokens: self.tokens,
        }
    }
}

impl<'a, S: Session> InstrFinalizer for BuiltinInstrSemantics<'a, S>
where
    S::Ident: 'static,
    S::StringRef: 'static,
    S::Span: 'static,
{
    type Next = TokenStreamSemantics<'a, S>;

    fn did_parse_instr(self) -> Self::Next {
        let args = self.state.args;
        let session = set_state!(self, InstrLineState::new().into());
        match self.state.mnemonic.item {
            BuiltinMnemonic::CpuInstr(cpu_instr) => {
                analyze_mnemonic((&cpu_instr, self.state.mnemonic.span), args, session)
                    .map_state(Into::into)
            }
            BuiltinMnemonic::Directive(directive) => directive::analyze_directive(
                (directive, self.state.mnemonic.span),
                self.state.label,
                args,
                session,
            ),
        }
    }
}

impl<'a, S: Session, T> Semantics<'a, S, T, S::Ident, S::StringRef, S::Span> {
    pub fn expect_const(
        &mut self,
        arg: Arg<S::Value, S::StringRef, S::Span>,
    ) -> Result<S::Value, ()> {
        match arg {
            Arg::Bare(DerefableArg::Const(value)) => Ok(value),
            Arg::Bare(DerefableArg::Symbol(_, span)) => {
                let keyword = self.session.strip_span(&span);
                self.session
                    .emit_diag(Message::KeywordInExpr { keyword }.at(span));
                Err(())
            }
            Arg::Error => Err(()),
            _ => unimplemented!(),
        }
    }

    pub fn define_symbol_with_params(
        &mut self,
        (name, span): (S::Ident, S::Span),
        expr: Arg<S::Value, S::StringRef, S::Span>,
    ) {
        if let Ok(value) = self.expect_const(expr) {
            let id = self.session.reloc_lookup(name, span.clone());
            self.session.define_symbol(id, span, value);
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

fn analyze_mnemonic<'a, S: Session>(
    name: (&Mnemonic, S::Span),
    args: BuiltinInstrArgs<S::Value, S::StringRef, S::Span>,
    mut session: TokenStreamSemantics<'a, S>,
) -> TokenStreamSemantics<'a, S> {
    let mut operands = Vec::new();
    for arg in args {
        let operand = cpu_instr::operand::analyze_operand(arg, name.0.context(), &mut session);
        operands.push(operand)
    }
    if let Ok(instruction) = cpu_instr::analyze_instruction(name, operands, &mut session) {
        session.session.emit_item(Item::CpuInstr(instruction))
    }
    session
}

#[cfg(test)]
mod tests {
    use crate::analyze::Literal::*;
    use crate::diag::{DiagnosticsEvent, Message, MockSpan};
    use crate::semantics::actions::tests::collect_semantic_actions;
    use crate::syntax::actions::*;
    use crate::syntax::actions::{ExprAtom::*, Operator::*};

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
