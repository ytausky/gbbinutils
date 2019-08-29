use self::cpu_instr::mnemonic::Mnemonic;
use self::directive::{BindingDirective, Directive, SimpleDirective};

use super::params::ResolveNames;
use super::resolve::{NameTable, StartScope};
use super::{BuiltinInstrArgs, InstrLineSemantics, Keyword, Label, TokenStreamSemantics};

use crate::analyze::reentrancy::ReentrancyActions;
use crate::object::builder::{Backend, Item};

use std::ops::DerefMut;

pub(super) mod cpu_instr;
pub(super) mod directive;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum BuiltinInstrMnemonic {
    Directive(Directive),
    CpuInstr(Mnemonic),
}

impl From<Directive> for BuiltinInstrMnemonic {
    fn from(directive: Directive) -> Self {
        BuiltinInstrMnemonic::Directive(directive)
    }
}

impl From<Mnemonic> for BuiltinInstrMnemonic {
    fn from(mnemonic: Mnemonic) -> Self {
        BuiltinInstrMnemonic::CpuInstr(mnemonic)
    }
}

pub(super) enum BuiltinInstr<S: ReentrancyActions> {
    Binding(
        (&'static BindingDirective, S::Span),
        Option<Label<S::Ident, S::Span>>,
    ),
    Directive((&'static SimpleDirective, S::Span)),
    CpuInstr((&'static Mnemonic, S::Span)),
}

impl<R: ReentrancyActions> BuiltinInstr<R> {
    pub fn new<N, B>(
        (mnemonic, span): (&'static BuiltinInstrMnemonic, R::Span),
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

    pub fn exec<N, B>(
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
            BuiltinInstr::Binding((binding, span), label) => directive::analyze_directive(
                (Directive::Binding(*binding), span),
                label,
                args,
                session,
            ),
            BuiltinInstr::Directive((simple, span)) => directive::analyze_directive(
                (Directive::Simple(*simple), span),
                None,
                args,
                session,
            ),
            BuiltinInstr::CpuInstr(mnemonic) => {
                analyze_mnemonic(mnemonic, args, session).map_state(Into::into)
            }
        }
    }
}

fn analyze_mnemonic<R: ReentrancyActions, N, B>(
    name: (&Mnemonic, R::Span),
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
        let (operand, returned_session) =
            cpu_instr::operand::analyze_operand(arg, name.0.context(), builder);
        session = returned_session;
        operands.push(operand)
    }
    if let Ok(instruction) = cpu_instr::analyze_instruction(name, operands, &mut session) {
        session.builder.emit_item(Item::CpuInstr(instruction))
    }
    session
}
