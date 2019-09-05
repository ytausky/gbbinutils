use self::cpu_instr::mnemonic::Mnemonic;
use self::directive::{BindingDirective, Directive, SimpleDirective};

use super::params::ResolveNames;
use super::resolve::{NameTable, StartScope};
use super::{BuiltinInstrArgs, Keyword, Label, TokenStreamSemantics};

use crate::analyze::reentrancy::ReentrancyActions;
use crate::diag::span::Spanned;
use crate::object::builder::{Backend, Item};

use std::ops::DerefMut;

pub(super) mod cpu_instr;
pub(super) mod directive;

pub(in crate::analyze) trait BuiltinInstrSet<R: ReentrancyActions>
where
    BuiltinInstr<&'static Self::Binding, &'static Self::NonBinding, R>: Dispatch<Self, R>,
{
    type Binding: 'static;
    type NonBinding: 'static;
    type Iter: Iterator<Item = &'static (&'static str, Keyword<Self::Binding, Self::NonBinding>)>;

    fn keywords() -> Self::Iter;
}

pub(in crate::analyze) struct DefaultBuiltinInstrSet;

impl<R: ReentrancyActions> BuiltinInstrSet<R> for DefaultBuiltinInstrSet {
    type Binding = BindingDirective;
    type NonBinding = UnboundBuiltinInstrMnemonic;
    type Iter = DefaultBuiltinInstrSetIter;

    fn keywords() -> Self::Iter {
        super::keywords::KEYWORDS.iter()
    }
}

type DefaultBuiltinInstrSetIter = std::slice::Iter<
    'static,
    (
        &'static str,
        Keyword<BindingDirective, UnboundBuiltinInstrMnemonic>,
    ),
>;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum BuiltinInstrMnemonic<L, U> {
    LabelBound(L),
    Unbound(U),
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum UnboundBuiltinInstrMnemonic {
    CpuInstr(Mnemonic),
    Directive(SimpleDirective),
}

impl From<Directive> for BuiltinInstrMnemonic<BindingDirective, UnboundBuiltinInstrMnemonic> {
    fn from(directive: Directive) -> Self {
        match directive {
            Directive::Binding(directive) => BuiltinInstrMnemonic::LabelBound(directive),
            Directive::Simple(directive) => {
                BuiltinInstrMnemonic::Unbound(UnboundBuiltinInstrMnemonic::Directive(directive))
            }
        }
    }
}

impl From<Mnemonic> for BuiltinInstrMnemonic<BindingDirective, UnboundBuiltinInstrMnemonic> {
    fn from(mnemonic: Mnemonic) -> Self {
        BuiltinInstrMnemonic::Unbound(UnboundBuiltinInstrMnemonic::CpuInstr(mnemonic))
    }
}

pub(in crate::analyze) enum BuiltinInstr<L, U, R: ReentrancyActions> {
    LabelBound(Option<Label<R::Ident, R::Span>>, Spanned<L, R::Span>),
    Unbound(Spanned<U, R::Span>),
}

pub(in crate::analyze) trait Dispatch<I: BuiltinInstrSet<R> + ?Sized, R: ReentrancyActions>
where
    BuiltinInstr<&'static I::Binding, &'static I::NonBinding, R>: Dispatch<I, R>,
{
    fn dispatch<N, B>(
        self,
        args: BuiltinInstrArgs<R::Ident, R::StringRef, R::Span>,
        session: TokenStreamSemantics<I, R, N, B>,
    ) -> TokenStreamSemantics<I, R, N, B>
    where
        N: DerefMut,
        N::Target: StartScope<R::Ident>
            + NameTable<
                R::Ident,
                Keyword = &'static Keyword<I::Binding, I::NonBinding>,
                MacroId = R::MacroId,
                SymbolId = B::SymbolId,
            >,
        B: Backend<R::Span>;
}

impl<R: ReentrancyActions> Dispatch<DefaultBuiltinInstrSet, R>
    for BuiltinInstr<&'static BindingDirective, &'static UnboundBuiltinInstrMnemonic, R>
{
    fn dispatch<N, B>(
        self,
        args: BuiltinInstrArgs<R::Ident, R::StringRef, R::Span>,
        session: TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B>,
    ) -> TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B>
    where
        N: DerefMut,
        N::Target: StartScope<R::Ident>
            + NameTable<
                R::Ident,
                Keyword = &'static Keyword<BindingDirective, UnboundBuiltinInstrMnemonic>,
                MacroId = R::MacroId,
                SymbolId = B::SymbolId,
            >,
        B: Backend<R::Span>,
    {
        match self {
            BuiltinInstr::LabelBound(label, mnemonic) => directive::analyze_directive(
                (Directive::Binding(*mnemonic.item), mnemonic.span),
                label,
                args,
                session,
            ),
            BuiltinInstr::Unbound(mnemonic) => match mnemonic.item {
                UnboundBuiltinInstrMnemonic::CpuInstr(cpu_instr) => {
                    analyze_mnemonic((cpu_instr, mnemonic.span), args, session)
                        .map_state(Into::into)
                }
                UnboundBuiltinInstrMnemonic::Directive(directive) => directive::analyze_directive(
                    (Directive::Simple(*directive), mnemonic.span),
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
    args: BuiltinInstrArgs<R::Ident, R::StringRef, R::Span>,
    mut session: TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B>,
) -> TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B>
where
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword<BindingDirective, UnboundBuiltinInstrMnemonic>,
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
