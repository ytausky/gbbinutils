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
use crate::object::builder::{Finish, PushOp};

use std::ops::DerefMut;

pub(super) mod builtin_instr;
mod label;
mod macro_instr;

pub(in crate::analyze) type InstrLineSemantics<R, N> = Session<R, N, InstrLineState<R>>;

pub(in crate::analyze) struct InstrLineState<S: ReentrancyActions> {
    pub label: Option<Label<S::Ident, S::Span>>,
}

impl<R, N> InstrLineActions<R::Ident, Literal<R::StringRef>, R::Span> for InstrLineSemantics<R, N>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword,
            MacroId = R::MacroId,
            SymbolId = R::SymbolId,
        >,
{
    type LabelActions = LabelSemantics<R, N>;
    type InstrActions = Self;

    fn will_parse_label(mut self, label: (R::Ident, R::Span)) -> Self::LabelActions {
        self = self.flush_label();
        self.map_line(|line| LabelState::new(line, label))
    }
}

impl<R, N> InstrActions<R::Ident, Literal<R::StringRef>, R::Span> for InstrLineSemantics<R, N>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword,
            MacroId = R::MacroId,
            SymbolId = R::SymbolId,
        >,
{
    type BuiltinInstrActions = BuiltinInstrSemantics<R, N>;
    type MacroInstrActions = MacroInstrSemantics<R, N>;
    type ErrorActions = Self;
    type LineFinalizer = TokenStreamSemantics<R, N>;

    fn will_parse_instr(
        mut self,
        ident: R::Ident,
        span: R::Span,
    ) -> InstrRule<Self::BuiltinInstrActions, Self::MacroInstrActions, Self> {
        match self.names.get(&ident) {
            Some(ResolvedName::Keyword(Keyword::BuiltinInstr(builtin))) => InstrRule::BuiltinInstr(
                self.map_line(|line| BuiltinInstrState::new(line, (builtin.clone(), span))),
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

impl<R, N> InstrLineSemantics<R, N>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword,
            MacroId = R::MacroId,
            SymbolId = R::SymbolId,
        >,
{
    pub fn flush_label(mut self) -> Self {
        if let Some(((label, span), _params)) = self.state.label.take() {
            self.names.start_scope(&label);
            let id = self.reloc_lookup(label, span.clone());
            let mut builder = self.reentrancy.define_symbol(id, span.clone());
            PushOp::<LocationCounter, _>::push_op(&mut builder, LocationCounter, span);
            let (reentrancy, ()) = builder.finish();
            self.reentrancy = reentrancy;
        }
        self
    }
}
