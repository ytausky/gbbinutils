use self::cpu_instr::mnemonic::Mnemonic;
use self::directive::Directive;

use super::keywords::{BuiltinMnemonic, FreeBuiltinMnemonic};
use super::resolve::{NameTable, StartScope};
use super::{
    BuiltinInstrArgs, BuiltinInstrSemantics, InstrLineState, Keyword, Label, TokenStreamSemantics,
};

use crate::analyze::reentrancy::ReentrancyActions;
use crate::diag::span::Spanned;
use crate::object::builder::{Backend, Item};

use std::ops::DerefMut;

pub(super) mod cpu_instr;
pub(super) mod directive;

impl From<Directive> for BuiltinMnemonic {
    fn from(directive: Directive) -> Self {
        match directive {
            Directive::Binding(directive) => BuiltinMnemonic::Binding(directive),
            Directive::Free(directive) => {
                BuiltinMnemonic::Free(FreeBuiltinMnemonic::Directive(directive))
            }
        }
    }
}

impl From<Mnemonic> for BuiltinMnemonic {
    fn from(mnemonic: Mnemonic) -> Self {
        BuiltinMnemonic::Free(FreeBuiltinMnemonic::CpuInstr(mnemonic))
    }
}

pub(in crate::analyze) enum BuiltinInstr<B, F, R: ReentrancyActions> {
    Binding(Option<Label<R::Ident, R::Span>>, Spanned<B, R::Span>),
    Free(Spanned<F, R::Span>),
}

pub(in crate::analyze) trait DispatchBuiltinInstrLine<R: ReentrancyActions, N, B> {
    fn dispatch_builtin_instr_line(self) -> TokenStreamSemantics<R, N, B>;
}

impl<R, N, B> DispatchBuiltinInstrLine<R, N, B> for BuiltinInstrSemantics<R, N, B>
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
    fn dispatch_builtin_instr_line(self) -> TokenStreamSemantics<R, N, B> {
        let instr = self.state.builtin_instr;
        let args = self.state.args;
        let session = set_state!(self, InstrLineState::new().into());
        match instr {
            BuiltinInstr::Binding(label, mnemonic) => directive::analyze_directive(
                (Directive::Binding(*mnemonic.item), mnemonic.span),
                label,
                args,
                session,
            ),
            BuiltinInstr::Free(mnemonic) => match mnemonic.item {
                FreeBuiltinMnemonic::CpuInstr(cpu_instr) => {
                    analyze_mnemonic((cpu_instr, mnemonic.span), args, session)
                        .map_state(Into::into)
                }
                FreeBuiltinMnemonic::Directive(directive) => directive::analyze_directive(
                    (Directive::Free(*directive), mnemonic.span),
                    None,
                    args,
                    session,
                ),
            },
        }
    }
}

fn analyze_mnemonic<R: ReentrancyActions, N, B>(
    name: (&Mnemonic, R::Span),
    args: BuiltinInstrArgs<B::Value, R::StringRef, R::Span>,
    mut session: TokenStreamSemantics<R, N, B>,
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
    let mut operands = Vec::new();
    for arg in args {
        let operand = cpu_instr::operand::analyze_operand(arg, name.0.context(), &mut session);
        operands.push(operand)
    }
    if let Ok(instruction) = cpu_instr::analyze_instruction(name, operands, &mut session) {
        session.builder.emit_item(Item::CpuInstr(instruction))
    }
    session
}
