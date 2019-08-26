pub(super) use self::builtin_instr::{BuiltinInstr, OperandSymbol};

use self::builtin_instr::cpu_instr::mnemonic::Mnemonic;
use self::builtin_instr::{BuiltinInstrSemantics, BuiltinInstrState};
use self::label::{LabelSemantics, LabelState};
use self::macro_instr::{MacroInstrSemantics, MacroInstrState};
use self::syntax::actions::{InstrActions, InstrLineActions, InstrRule};

use super::diag::span::StripSpan;
use super::diag::{EmitDiag, Message};
use super::params::RelocLookup;
use super::syntax;
use super::{Keyword, Label, Literal, ReentrancyActions, Session, TokenStreamSemantics};

use crate::analyze::resolve::{NameTable, ResolvedName, StartScope};
use crate::expr::LocationCounter;
use crate::object::builder::{Backend, Finish, PushOp};

use std::ops::DerefMut;

pub(super) mod builtin_instr;
mod label;
mod macro_instr;

pub(in crate::analyze) type InstrLineSemantics<R, N, B> = Session<R, N, B, InstrLineState<R>>;

pub(in crate::analyze) struct InstrLineState<S: ReentrancyActions> {
    pub label: Option<Label<S::Ident, S::Span>>,
}

impl<R, N, B> InstrLineActions<R::Ident, Literal<R::StringRef>, R::Span>
    for InstrLineSemantics<R, N, B>
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
    type LabelActions = LabelSemantics<R, N, B>;
    type InstrActions = Self;

    fn will_parse_label(mut self, label: (R::Ident, R::Span)) -> Self::LabelActions {
        self = self.flush_label();
        self.map_state(|line| LabelState::new(line, label))
    }
}

impl<R, N, B> InstrActions<R::Ident, Literal<R::StringRef>, R::Span> for InstrLineSemantics<R, N, B>
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
    type BuiltinInstrActions = BuiltinInstrSemantics<R, N, B>;
    type MacroInstrActions = MacroInstrSemantics<R, N, B>;
    type ErrorActions = Self;
    type LineFinalizer = TokenStreamSemantics<R, N, B>;

    fn will_parse_instr(
        mut self,
        ident: R::Ident,
        span: R::Span,
    ) -> InstrRule<Self::BuiltinInstrActions, Self::MacroInstrActions, Self> {
        match self.names.resolve_name(&ident) {
            Some(ResolvedName::Keyword(Keyword::BuiltinInstr(builtin))) => InstrRule::BuiltinInstr(
                self.map_state(|line| BuiltinInstrState::new(line, (builtin.clone(), span))),
            ),
            Some(ResolvedName::Keyword(Keyword::Operand(_))) => unimplemented!(),
            Some(ResolvedName::Macro(id)) => {
                self = self.flush_label();
                InstrRule::MacroInstr(set_state!(
                    self,
                    MacroInstrState::new(self.state, (id, span))
                ))
            }
            Some(ResolvedName::Symbol(_)) => {
                let name = self.strip_span(&span);
                self.emit_diag(Message::CannotUseSymbolNameAsMacroName { name }.at(span));
                InstrRule::Error(self)
            }
            None => {
                let name = self.strip_span(&span);
                self.emit_diag(Message::UndefinedMacro { name }.at(span));
                InstrRule::Error(self)
            }
        }
    }
}

impl<S: ReentrancyActions> InstrLineState<S> {
    pub fn new() -> Self {
        Self { label: None }
    }
}

impl<R, N, B> InstrLineSemantics<R, N, B>
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
    pub fn flush_label(mut self) -> Self {
        if let Some(((label, span), _params)) = self.state.label.take() {
            self.names.start_scope(&label);
            let id = self.reloc_lookup(label, span.clone());
            let mut builder = self.builder.define_symbol(id, span.clone());
            PushOp::<LocationCounter, _>::push_op(&mut builder, LocationCounter, span);
            let (builder, ()) = builder.finish();
            self.builder = builder;
        }
        self
    }
}
