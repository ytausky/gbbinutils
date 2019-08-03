use self::builtin_instr::cpu_instr::mnemonic::*;
use self::builtin_instr::{BuiltinInstr, BuiltinInstrSemantics};
use self::builtin_instr::{BuiltinInstr::*, Directive::*};
use self::label::LabelSemantics;
use self::macro_instr::MacroInstrSemantics;
use self::syntax::{InstrActions, InstrLineActions, InstrRule};

use super::diag::span::StripSpan;
use super::diag::{EmitDiag, Message};
use super::params::RelocLookup;
use super::resolve::ResolvedIdent;
use super::syntax;
use super::{Label, Literal, SemanticState, Session, TokenStreamSemantics};

use crate::analysis::backend::{Finish, PushOp};
use crate::model::LocationCounter;

mod builtin_instr;
mod label;
mod macro_instr;

pub(in crate::analysis) type InstrLineSemantics<S> = SemanticState<InstrLineState<S>, S>;

pub(in crate::analysis) struct InstrLineState<S: Session> {
    pub label: Option<Label<S::Ident, S::Span>>,
}

impl<S: Session> InstrLineActions<S::Ident, Literal<S::StringRef>, S::Span>
    for InstrLineSemantics<S>
where
    S::Ident: AsRef<str>,
{
    type LabelActions = LabelSemantics<S>;
    type InstrActions = Self;

    fn will_parse_label(mut self, label: (S::Ident, S::Span)) -> Self::LabelActions {
        self = self.define_label_if_present();
        LabelSemantics::new(self, label)
    }
}

impl<S: Session> InstrActions<S::Ident, Literal<S::StringRef>, S::Span> for InstrLineSemantics<S>
where
    S::Ident: AsRef<str>,
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
        match KEYS
            .iter()
            .find(|(spelling, _)| spelling.eq_ignore_ascii_case(ident.as_ref()))
            .map(|(_, entry)| entry)
        {
            Some(KeyEntry::BuiltinInstr(command)) => {
                InstrRule::BuiltinInstr(BuiltinInstrSemantics::new(self, (command.clone(), span)))
            }
            None => match self.session.get(&ident) {
                Some(ResolvedIdent::Macro(id)) => {
                    self = self.define_label_if_present();
                    InstrRule::MacroInstr(MacroInstrSemantics::new(self, (id, span)))
                }
                Some(ResolvedIdent::Backend(_)) => {
                    let name = self.strip_span(&span);
                    self.emit_diag(Message::CannotUseSymbolNameAsMacroName { name }.at(span));
                    InstrRule::Error(self)
                }
                None => {
                    let name = self.strip_span(&span);
                    self.emit_diag(Message::UndefinedMacro { name }.at(span));
                    InstrRule::Error(self)
                }
            },
        }
    }
}

impl<S: Session> InstrLineSemantics<S> {
    pub fn new(session: S) -> Self {
        Self {
            line: InstrLineState { label: None },
            session,
        }
    }

    pub fn define_label_if_present(mut self) -> Self {
        if let Some(((label, span), _params)) = self.line.label.take() {
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

#[derive(Clone)]
enum KeyEntry {
    BuiltinInstr(BuiltinInstr),
}

const KEYS: &[(&str, KeyEntry)] = &[
    ("adc", KeyEntry::BuiltinInstr(Mnemonic(ADC))),
    ("add", KeyEntry::BuiltinInstr(Mnemonic(ADD))),
    ("and", KeyEntry::BuiltinInstr(Mnemonic(AND))),
    ("bit", KeyEntry::BuiltinInstr(Mnemonic(BIT))),
    ("call", KeyEntry::BuiltinInstr(Mnemonic(CALL))),
    ("cp", KeyEntry::BuiltinInstr(Mnemonic(CP))),
    ("cpl", KeyEntry::BuiltinInstr(Mnemonic(CPL))),
    ("daa", KeyEntry::BuiltinInstr(Mnemonic(DAA))),
    ("db", KeyEntry::BuiltinInstr(Directive(Db))),
    ("dec", KeyEntry::BuiltinInstr(Mnemonic(DEC))),
    ("di", KeyEntry::BuiltinInstr(Mnemonic(DI))),
    ("ds", KeyEntry::BuiltinInstr(Directive(Ds))),
    ("dw", KeyEntry::BuiltinInstr(Directive(Dw))),
    ("ei", KeyEntry::BuiltinInstr(Mnemonic(EI))),
    ("equ", KeyEntry::BuiltinInstr(Directive(Equ))),
    ("halt", KeyEntry::BuiltinInstr(Mnemonic(HALT))),
    ("inc", KeyEntry::BuiltinInstr(Mnemonic(INC))),
    ("include", KeyEntry::BuiltinInstr(Directive(Include))),
    ("jp", KeyEntry::BuiltinInstr(Mnemonic(JP))),
    ("jr", KeyEntry::BuiltinInstr(Mnemonic(JR))),
    ("ld", KeyEntry::BuiltinInstr(Mnemonic(LD))),
    ("ldhl", KeyEntry::BuiltinInstr(Mnemonic(LDHL))),
    ("macro", KeyEntry::BuiltinInstr(Directive(Macro))),
    ("nop", KeyEntry::BuiltinInstr(Mnemonic(NOP))),
    ("or", KeyEntry::BuiltinInstr(Mnemonic(OR))),
    ("org", KeyEntry::BuiltinInstr(Directive(Org))),
    ("pop", KeyEntry::BuiltinInstr(Mnemonic(POP))),
    ("push", KeyEntry::BuiltinInstr(Mnemonic(PUSH))),
    ("res", KeyEntry::BuiltinInstr(Mnemonic(RES))),
    ("ret", KeyEntry::BuiltinInstr(Mnemonic(RET))),
    ("reti", KeyEntry::BuiltinInstr(Mnemonic(RETI))),
    ("rl", KeyEntry::BuiltinInstr(Mnemonic(RL))),
    ("rla", KeyEntry::BuiltinInstr(Mnemonic(RLA))),
    ("rlc", KeyEntry::BuiltinInstr(Mnemonic(RLC))),
    ("rlca", KeyEntry::BuiltinInstr(Mnemonic(RLCA))),
    ("rr", KeyEntry::BuiltinInstr(Mnemonic(RR))),
    ("rra", KeyEntry::BuiltinInstr(Mnemonic(RRA))),
    ("rrc", KeyEntry::BuiltinInstr(Mnemonic(RRC))),
    ("rrca", KeyEntry::BuiltinInstr(Mnemonic(RRCA))),
    ("rst", KeyEntry::BuiltinInstr(Mnemonic(RST))),
    ("sbc", KeyEntry::BuiltinInstr(Mnemonic(SBC))),
    ("section", KeyEntry::BuiltinInstr(Directive(Section))),
    ("set", KeyEntry::BuiltinInstr(Mnemonic(SET))),
    ("sla", KeyEntry::BuiltinInstr(Mnemonic(SLA))),
    ("sra", KeyEntry::BuiltinInstr(Mnemonic(SRA))),
    ("srl", KeyEntry::BuiltinInstr(Mnemonic(SRL))),
    ("stop", KeyEntry::BuiltinInstr(Mnemonic(STOP))),
    ("sub", KeyEntry::BuiltinInstr(Mnemonic(SUB))),
    ("swap", KeyEntry::BuiltinInstr(Mnemonic(SWAP))),
    ("xor", KeyEntry::BuiltinInstr(Mnemonic(XOR))),
];
