use self::label::{LabelSemantics, LabelState};
use self::macro_instr::{MacroInstrSemantics, MacroInstrState};

use super::{Keyword, Semantics, TokenStreamSemantics};

use crate::diag::span::WithSpan;
use crate::diag::Message;
use crate::expr::{Atom, Expr, ExprOp};
use crate::semantics::*;
use crate::session::resolve::{NameTable, ResolvedName};
use crate::syntax::actions::{InstrContext, InstrLineContext, InstrRule};

mod builtin_instr;
mod label;
mod macro_instr;

impl<'a, 'b, S: Session> InstrLineContext for InstrLineSemantics<'a, 'b, S>
where
    S::Ident: 'static,
    S::StringRef: 'static,
    S::Span: 'static,
{
    type LabelContext = LabelSemantics<'a, 'b, S>;
    type InstrContext = Self;

    fn will_parse_label(mut self, label: (S::Ident, S::Span)) -> Self::LabelContext {
        self = self.flush_label();
        self.map_state(|line| LabelState::new(line, label))
    }
}

impl<'a, 'b, S: Session> InstrContext for InstrLineSemantics<'a, 'b, S>
where
    S::Ident: 'static,
    S::StringRef: 'static,
    S::Span: 'static,
{
    type BuiltinInstrContext = BuiltinInstrSemantics<'a, 'b, S>;
    type MacroInstrContext = MacroInstrSemantics<'a, 'b, S>;
    type ErrorContext = Self;
    type LineFinalizer = TokenStreamSemantics<'a, 'b, S>;

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

impl<'a, 'b, S: Session> InstrLineSemantics<'a, 'b, S> {
    pub fn flush_label(mut self) -> Self {
        if let Some(((label, span), _params)) = self.state.label.take() {
            self.session.start_scope(&label);
            let id = self.reloc_lookup(label, span.clone());
            self.session.define_symbol(
                id,
                span.clone(),
                Expr(vec![ExprOp::Atom(Atom::Location).with_span(span)]),
            );
        }
        self
    }
}

impl<'a, 'b, S, T, I, R, Z> Semantics<'a, 'b, S, T, I, R, Z>
where
    S: AllocSymbol<Z> + NameTable<I> + Diagnostics<Z>,
    Z: Clone,
{
    fn reloc_lookup(&mut self, name: I, span: Z) -> S::SymbolId {
        match self.session.resolve_name(&name) {
            Some(ResolvedName::Keyword(_)) => unimplemented!(),
            Some(ResolvedName::Symbol(id)) => id,
            None => {
                let id = self.session.alloc_symbol(span);
                self.session
                    .define_name(name, ResolvedName::Symbol(id.clone()));
                id
            }
            Some(ResolvedName::Macro(_)) => {
                self.session
                    .emit_diag(Message::MacroNameInExpr.at(span.clone()));
                self.session.alloc_symbol(span)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::diag::{DiagnosticsEvent, Message, MockSpan};
    use crate::semantics::actions::tests::*;
    use crate::session::builder::mock::MockSymbolId;
    use crate::syntax::actions::*;

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
