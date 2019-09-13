use self::label::{LabelSemantics, LabelState};
use self::macro_instr::{MacroInstrSemantics, MacroInstrState};

use super::token_line::TokenContext;
use super::{Keyword, ReentrancyActions, Session, TokenStreamSemantics};

use crate::analyze::semantics::builtin_instr::DispatchBuiltinInstrLine;
use crate::analyze::semantics::params::RelocLookup;
use crate::analyze::semantics::resolve::{NameTable, ResolvedName, StartScope};
use crate::analyze::semantics::*;
use crate::analyze::syntax::actions::{InstrActions, InstrLineActions, InstrRule};
use crate::analyze::Literal;
use crate::diag::span::{StripSpan, WithSpan};
use crate::diag::{EmitDiag, Message};
use crate::expr::LocationCounter;
use crate::object::builder::{Backend, Finish, PushOp};

use std::ops::DerefMut;

mod builtin_instr;
mod label;
mod macro_instr;

impl<I, R, N, B> InstrLineActions<R::Ident, Literal<R::StringRef>, R::Span>
    for InstrLineSemantics<I, R, N, B>
where
    I: BuiltinInstrSet<R>,
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword<I::Binding, I::Free>,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: Backend<R::Span>,
    BuiltinInstrSemantics<I, R, N, B>: DispatchBuiltinInstrLine<I, R, N, B>,
    TokenLineContext<R::Ident, R::StringRef, R::Span>: TokenContext<I, R>,
{
    type LabelActions = LabelSemantics<I, R, N, B>;
    type InstrActions = Self;

    fn will_parse_label(mut self, label: (R::Ident, R::Span)) -> Self::LabelActions {
        self = self.flush_label();
        self.map_state(|line| LabelState::new(line, label))
    }
}

impl<I, R, N, B> InstrActions<R::Ident, Literal<R::StringRef>, R::Span>
    for InstrLineSemantics<I, R, N, B>
where
    I: BuiltinInstrSet<R>,
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword<I::Binding, I::Free>,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: Backend<R::Span>,
    BuiltinInstrSemantics<I, R, N, B>: DispatchBuiltinInstrLine<I, R, N, B>,
    TokenLineContext<R::Ident, R::StringRef, R::Span>: TokenContext<I, R>,
{
    type BuiltinInstrActions = BuiltinInstrSemantics<I, R, N, B>;
    type MacroInstrActions = MacroInstrSemantics<I, R, N, B>;
    type ErrorActions = Self;
    type LineFinalizer = TokenStreamSemantics<I, R, N, B>;

    fn will_parse_instr(
        mut self,
        ident: R::Ident,
        span: R::Span,
    ) -> InstrRule<Self::BuiltinInstrActions, Self::MacroInstrActions, Self> {
        match self.names.resolve_name(&ident) {
            Some(ResolvedName::Keyword(Keyword::BuiltinMnemonic(mnemonic))) => {
                let builtin_instr = match mnemonic {
                    BuiltinMnemonic::Binding(directive) => {
                        BuiltinInstr::Binding(self.state.label, directive.with_span(span))
                    }
                    BuiltinMnemonic::Free(unbound) => {
                        self = self.flush_label();
                        BuiltinInstr::Free(unbound.with_span(span))
                    }
                };
                InstrRule::BuiltinInstr(set_state!(self, BuiltinInstrState::new(builtin_instr)))
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

impl<I, R, N, B> InstrLineSemantics<I, R, N, B>
where
    I: BuiltinInstrSet<R>,
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword<I::Binding, I::Free>,
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
