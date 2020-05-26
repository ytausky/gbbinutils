use self::label::{LabelSemantics, LabelState};
use self::macro_instr::{MacroInstrSemantics, MacroInstrState};

use super::{Keyword, ReentrancyActions, Semantics, TokenStreamSemantics};

use crate::analyze::semantics::params::RelocLookup;
use crate::analyze::semantics::resolve::{NameTable, ResolvedName, StartScope};
use crate::analyze::semantics::*;
use crate::analyze::syntax::actions::{InstrContext, InstrLineContext, InstrRule};
use crate::diag::span::{StripSpan, WithSpan};
use crate::diag::{EmitDiag, Message};
use crate::expr::LocationCounter;
use crate::object::builder::{Backend, Finish, PushOp};

use std::ops::DerefMut;

mod builtin_instr;
mod label;
mod macro_instr;

impl<'a, R, N, B> InstrLineContext for InstrLineSemantics<'a, R, N, B>
where
    R: Meta,
    R::Ident: 'static,
    R::StringRef: 'static,
    R::Span: 'static,
    Core<R, N, B>: ReentrancyActions<
        Ident = R::Ident,
        StringRef = R::StringRef,
        Span = R::Span,
        MacroId = R::MacroId,
    >,
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
    type LabelContext = LabelSemantics<'a, R, N, B>;
    type InstrContext = Self;

    fn will_parse_label(mut self, label: (R::Ident, R::Span)) -> Self::LabelContext {
        self = self.flush_label();
        self.map_state(|line| LabelState::new(line, label))
    }
}

impl<'a, R, N, B> InstrContext for InstrLineSemantics<'a, R, N, B>
where
    R: Meta,
    R::Ident: 'static,
    R::StringRef: 'static,
    R::Span: 'static,
    Core<R, N, B>: ReentrancyActions<
        Ident = R::Ident,
        StringRef = R::StringRef,
        Span = R::Span,
        MacroId = R::MacroId,
    >,
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
    type BuiltinInstrContext = BuiltinInstrSemantics<'a, R, N, B>;
    type MacroInstrContext = MacroInstrSemantics<'a, R, N, B>;
    type ErrorContext = Self;
    type LineFinalizer = TokenStreamSemantics<'a, R, N, B>;

    fn will_parse_instr(
        mut self,
        ident: R::Ident,
        span: R::Span,
    ) -> InstrRule<Self::BuiltinInstrContext, Self::MacroInstrContext, Self> {
        match self.core.names.resolve_name(&ident) {
            Some(ResolvedName::Keyword(Keyword::BuiltinMnemonic(mnemonic))) => {
                if !mnemonic.binds_to_label() {
                    self = self.flush_label();
                }
                InstrRule::BuiltinInstr(set_state!(
                    self,
                    BuiltinInstrState::new(self.state.label, mnemonic.clone().with_span(span))
                ))
            }
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
            Some(ResolvedName::Keyword(Keyword::Operand(_))) | None => {
                let name = self.strip_span(&span);
                self.emit_diag(Message::NotAMnemonic { name }.at(span));
                InstrRule::Error(self)
            }
        }
    }
}

impl<'a, R, N, B> InstrLineSemantics<'a, R, N, B>
where
    R: Meta,
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
            self.core.names.start_scope(&label);
            let id = self.reloc_lookup(label, span.clone());
            let mut builder = self.core.builder.build_const();
            PushOp::<LocationCounter, _>::push_op(&mut builder, LocationCounter, span.clone());
            let (mut builder, expr) = builder.finish();
            builder.define_symbol(id, span, expr.unwrap());
            self.core.builder = builder;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analyze::semantics::actions::tests::*;
    use crate::analyze::syntax::actions::*;
    use crate::diag::{DiagnosticsEvent, Message, MockSpan};
    use crate::object::builder::mock::MockSymbolId;

    #[test]
    fn diagnose_unknown_mnemonic() {
        let name = "unknown";
        let log = collect_semantic_actions::<_, MockSpan<_>>(|session| {
            session
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr(name.into(), name.into())
                .error()
                .unwrap()
                .did_parse_instr()
                .did_parse_line("eol".into())
                .act_on_eos("eos".into())
        });
        assert_eq!(
            log,
            [DiagnosticsEvent::EmitDiag(
                Message::NotAMnemonic { name: name.into() }
                    .at(name.into())
                    .into()
            )
            .into()]
        )
    }

    #[test]
    fn diagnose_operand_as_mnemonic() {
        let name = "HL";
        let log = collect_semantic_actions::<_, MockSpan<_>>(|session| {
            session
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr(name.into(), name.into())
                .error()
                .unwrap()
                .did_parse_instr()
                .did_parse_line("eol".into())
                .act_on_eos("eos".into())
        });
        assert_eq!(
            log,
            [DiagnosticsEvent::EmitDiag(
                Message::NotAMnemonic { name: name.into() }
                    .at(name.into())
                    .into()
            )
            .into()]
        )
    }

    #[test]
    fn diagnose_symbol_as_mnemonic() {
        let name = "symbol";
        let log = log_with_predefined_names::<_, _, MockSpan<_>>(
            vec![(name.into(), ResolvedName::Symbol(MockSymbolId(42)))],
            |session| {
                session
                    .will_parse_line()
                    .into_instr_line()
                    .will_parse_instr(name.into(), name.into())
                    .error()
                    .unwrap()
                    .did_parse_line("eol".into())
                    .act_on_eos("eos".into())
            },
        );
        assert_eq!(
            log,
            [DiagnosticsEvent::EmitDiag(
                Message::CannotUseSymbolNameAsMacroName { name: name.into() }
                    .at(name.into())
                    .into()
            )
            .into()]
        )
    }
}
