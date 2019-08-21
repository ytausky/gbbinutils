pub(super) use self::builtin_instr::{BuiltinInstr, OperandSymbol};

use self::builtin_instr::cpu_instr::mnemonic::Mnemonic;
use self::builtin_instr::{BuiltinInstrSemantics, BuiltinInstrState};
use self::label::{LabelSemantics, LabelState};
use self::macro_instr::{MacroInstrSemantics, MacroInstrState};
use self::syntax::actions::{InstrActions, InstrLineActions, InstrRule};

use super::diag::span::StripSpan;
use super::diag::{EmitDiag, Message};
use super::params::RelocLookup;
use super::resolve::ResolvedName;
use super::syntax;
use super::{Keyword, Label, Literal, SemanticActions, Session, TokenStreamSemantics};

use crate::expr::LocationCounter;
use crate::object::builder::{Finish, PushOp};

pub(super) mod builtin_instr;
mod label;
mod macro_instr;

pub(in crate::analyze) type InstrLineSemantics<S> = SemanticActions<InstrLineState<S>, S>;

pub(in crate::analyze) struct InstrLineState<S: Session> {
    pub label: Option<Label<S::Ident, S::Span>>,
}

impl<S> InstrLineActions<S::Ident, Literal<S::StringRef>, S::Span> for InstrLineSemantics<S>
where
    S: Session<Keyword = &'static Keyword>,
{
    type LabelActions = LabelSemantics<S>;
    type InstrActions = Self;

    fn will_parse_label(mut self, label: (S::Ident, S::Span)) -> Self::LabelActions {
        self = self.define_label_if_present();
        self.map_line(|line| LabelState::new(line, label))
    }
}

impl<S> InstrActions<S::Ident, Literal<S::StringRef>, S::Span> for InstrLineSemantics<S>
where
    S: Session<Keyword = &'static Keyword>,
{
    type BuiltinInstrActions = BuiltinInstrSemantics<S>;
    type MacroInstrActions = MacroInstrSemantics<S>;
    type ErrorActions = Self;
    type LineFinalizer = TokenStreamSemantics<S>;

    fn will_parse_instr(
        mut self,
        ident: S::Ident,
        span: S::Span,
    ) -> InstrRule<Self::BuiltinInstrActions, Self::MacroInstrActions, Self> {
        match self.session.get(&ident) {
            Some(ResolvedName::Keyword(Keyword::BuiltinInstr(builtin))) => InstrRule::BuiltinInstr(
                self.map_line(|line| BuiltinInstrState::new(line, (builtin.clone(), span))),
            ),
            Some(ResolvedName::Keyword(Keyword::Operand(_))) => unimplemented!(),
            Some(ResolvedName::Macro(id)) => {
                self = self.define_label_if_present();
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

impl<S: Session> InstrLineState<S> {
    pub fn new() -> Self {
        Self { label: None }
    }
}

impl<S: Session> InstrLineSemantics<S> {
    pub fn define_label_if_present(mut self) -> Self {
        if let Some(((label, span), _params)) = self.state.label.take() {
            self.session.start_scope(&label);
            let id = self.session.reloc_lookup(label, span.clone());
            let mut builder = self.session.define_symbol(id, span.clone());
            PushOp::<LocationCounter, _>::push_op(&mut builder, LocationCounter, span);
            let (session, ()) = builder.finish();
            self.session = session;
        }
        self
    }
}
