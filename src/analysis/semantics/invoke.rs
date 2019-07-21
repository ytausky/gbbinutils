use super::StmtActions;

use crate::analysis::resolve::ResolvedIdent;
use crate::analysis::session::{MacroArgs, Session};
use crate::analysis::syntax::{MacroCallContext, TokenSeqContext};
use crate::analysis::{SemanticToken, TokenSeq};
use crate::diag::Message;

pub(in crate::analysis) struct MacroCallActions<S: Session> {
    parent: StmtActions<S>,
    name: (S::Ident, S::Span),
    args: MacroArgs<S::Ident, S::StringRef, S::Span>,
}

impl<S: Session> MacroCallActions<S> {
    pub fn new(parent: StmtActions<S>, name: (S::Ident, S::Span)) -> MacroCallActions<S> {
        MacroCallActions {
            parent,
            name,
            args: (Vec::new(), Vec::new()),
        }
    }

    fn push_arg(&mut self, arg: TokenSeq<S::Ident, S::StringRef, S::Span>) {
        self.args.0.push(arg.0);
        self.args.1.push(arg.1);
    }
}

delegate_diagnostics! {
    {S: Session}, MacroCallActions<S>, {parent}, StmtActions<S>, S::Span
}

impl<S: Session> MacroCallContext<S::Span> for MacroCallActions<S> {
    type Token = SemanticToken<S::Ident, S::StringRef>;
    type Parent = StmtActions<S>;
    type MacroArgContext = MacroArgContext<S>;

    fn enter_macro_arg(self) -> Self::MacroArgContext {
        MacroArgContext::new(self)
    }

    fn exit(self) -> Self::Parent {
        let Self {
            mut parent,
            name: (name, span),
            args,
        } = self;
        parent.parent.with_session(|mut session| {
            let stripped = session.strip_span(&span);
            let result = match session.get(&name) {
                Some(ResolvedIdent::Macro(id)) => Ok((id, span)),
                Some(ResolvedIdent::Backend(_)) => {
                    Err(Message::CannotUseSymbolNameAsMacroName { name: stripped }.at(span))
                }
                None => Err(Message::UndefinedMacro { name: stripped }.at(span)),
            };
            match result {
                Ok(spanned_id) => session = session.call_macro(spanned_id, args),
                Err(diag) => session.emit_diag(diag),
            }
            (session, ())
        });
        parent
    }
}

pub(in crate::analysis) struct MacroArgContext<S: Session> {
    tokens: TokenSeq<S::Ident, S::StringRef, S::Span>,
    parent: MacroCallActions<S>,
}

impl<S: Session> MacroArgContext<S> {
    fn new(parent: MacroCallActions<S>) -> MacroArgContext<S> {
        MacroArgContext {
            tokens: (Vec::new(), Vec::new()),
            parent,
        }
    }
}

delegate_diagnostics! {
    {S: Session}, MacroArgContext<S>, {parent}, MacroCallActions<S>, S::Span
}

impl<S: Session> TokenSeqContext<S::Span> for MacroArgContext<S> {
    type Token = SemanticToken<S::Ident, S::StringRef>;
    type Parent = MacroCallActions<S>;

    fn push_token(&mut self, token: (Self::Token, S::Span)) {
        self.tokens.0.push(token.0);
        self.tokens.1.push(token.1);
    }

    fn exit(mut self) -> Self::Parent {
        self.parent.push_arg(self.tokens);
        self.parent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::semantics::tests::*;
    use crate::analysis::semantics::{Operand, SemanticActions};
    use crate::analysis::session::{MockMacroId, SessionEvent};
    use crate::analysis::syntax::{FileContext, StmtContext, Token};
    use crate::analysis::Literal;
    use crate::diag::{DiagnosticsEvent, Merge, MockSpan};
    use crate::log::with_log;

    use std::fmt::Debug;

    #[test]
    fn call_nullary_macro() {
        let name = "my_macro";
        let macro_id = MockMacroId(0);
        let log = log_with_predefined_names(
            vec![(name.into(), ResolvedIdent::Macro(macro_id))],
            |actions| {
                let call = actions
                    .enter_unlabeled_stmt()
                    .enter_macro_call((name.into(), ()));
                call.exit().exit()
            },
        );
        assert_eq!(
            log,
            [SessionEvent::InvokeMacro(macro_id, Vec::new()).into()]
        )
    }

    #[test]
    fn call_unary_macro() {
        let name = "my_macro";
        let arg_token = Token::Literal(Literal::Operand(Operand::A));
        let macro_id = MockMacroId(0);
        let log = log_with_predefined_names(
            vec![(name.into(), ResolvedIdent::Macro(macro_id))],
            |actions| {
                let mut call = actions
                    .enter_unlabeled_stmt()
                    .enter_macro_call((name.into(), ()));
                call = {
                    let mut arg = call.enter_macro_arg();
                    arg.push_token((arg_token.clone(), ()));
                    arg.exit()
                };
                call.exit().exit()
            },
        );
        assert_eq!(
            log,
            [SessionEvent::InvokeMacro(macro_id, vec![vec![arg_token]]).into()]
        )
    }

    #[test]
    fn diagnose_undefined_macro() {
        let name = "my_macro";
        let span = name;
        let log = collect_semantic_actions::<_, MockSpan<_>>(|actions| {
            actions
                .enter_unlabeled_stmt()
                .enter_macro_call((name.into(), span.into()))
                .exit()
                .exit()
        });
        assert_eq!(
            log,
            [DiagnosticsEvent::EmitDiag(
                Message::UndefinedMacro { name: span.into() }
                    .at(span.into())
                    .into()
            )
            .into()]
        );
    }

    #[test]
    fn diagnose_symbol_name_in_macro_call() {
        let name = "symbol";
        let as_macro = MockSpan::from("as_macro");
        let log =
            log_with_predefined_names(vec![(name.into(), ResolvedIdent::Backend(0))], |actions| {
                actions
                    .enter_unlabeled_stmt()
                    .enter_macro_call((name.into(), as_macro.clone()))
                    .exit()
                    .exit()
            });
        assert_eq!(
            log,
            [DiagnosticsEvent::EmitDiag(
                Message::CannotUseSymbolNameAsMacroName {
                    name: as_macro.clone()
                }
                .at(as_macro)
                .into()
            )
            .into()]
        )
    }

    fn log_with_predefined_names<I, F, S>(entries: I, f: F) -> Vec<TestOperation<S>>
    where
        I: IntoIterator<Item = (String, ResolvedIdent<usize, MockMacroId>)>,
        F: FnOnce(TestSemanticActions<S>) -> TestSemanticActions<S>,
        S: Clone + Debug + Merge,
    {
        with_log(|log| {
            f(SemanticActions::new(MockSession::with_predefined_names(
                log, entries,
            )));
        })
    }
}
