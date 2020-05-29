use self::label::{LabelSemantics, LabelState};
use self::macro_instr::{MacroInstrSemantics, MacroInstrState};

use super::{Keyword, Semantics, TokenStreamSemantics};

use crate::analyze::semantics::params::RelocLookup;
use crate::analyze::semantics::session::resolve::{NameTable, ResolvedName, StartScope};
use crate::analyze::semantics::*;
use crate::analyze::syntax::actions::{InstrContext, InstrLineContext, InstrRule};
use crate::diag::span::WithSpan;
use crate::diag::Message;
use crate::expr::LocationCounter;
use crate::object::builder::{Finish, PushOp};

mod builtin_instr;
mod label;
mod macro_instr;

impl<'a, S: Session> InstrLineContext for InstrLineSemantics<'a, S>
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
    type LabelContext = LabelSemantics<'a, S>;
    type InstrContext = Self;

    fn will_parse_label(mut self, label: (S::Ident, S::Span)) -> Self::LabelContext {
        self = self.flush_label();
        self.map_state(|line| LabelState::new(line, label))
    }
}

impl<'a, S: Session> InstrContext for InstrLineSemantics<'a, S>
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
    type BuiltinInstrContext = BuiltinInstrSemantics<'a, S>;
    type MacroInstrContext = MacroInstrSemantics<'a, S>;
    type ErrorContext = Self;
    type LineFinalizer = TokenStreamSemantics<'a, S>;

    fn will_parse_instr(
        mut self,
        ident: S::Ident,
        span: S::Span,
    ) -> InstrRule<Self::BuiltinInstrContext, Self::MacroInstrContext, Self> {
        match self.session.resolve_name(&ident) {
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
                let name = self.session.strip_span(&span);
                self.session
                    .emit_diag(Message::CannotUseSymbolNameAsMacroName { name }.at(span));
                InstrRule::Error(self)
            }
            Some(ResolvedName::Keyword(Keyword::Operand(_))) | None => {
                let name = self.session.strip_span(&span);
                self.session
                    .emit_diag(Message::NotAMnemonic { name }.at(span));
                InstrRule::Error(self)
            }
        }
    }
}

impl<'a, S: Session> InstrLineSemantics<'a, S> {
    pub fn flush_label(mut self) -> Self {
        if let Some(((label, span), _params)) = self.state.label.take() {
            self.session.start_scope(&label);
            let id = self.session.reloc_lookup(label, span.clone());
            let mut builder = self.session.build_const();
            PushOp::<LocationCounter, _>::push_op(&mut builder, LocationCounter, span.clone());
            let (mut session, expr) = builder.finish();
            session.define_symbol(id, span, expr.unwrap());
            self.session = session;
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
