use super::SemanticActions;

use crate::analysis::session::{MacroArgs, Session};
use crate::analysis::syntax::{InstrEndContext, MacroCallContext, TokenSeqContext};
use crate::analysis::{SemanticToken, TokenSeq};

pub(in crate::analysis) struct MacroCallActions<S: Session> {
    parent: SemanticActions<S>,
    name: (S::MacroEntry, S::Span),
    args: MacroArgs<S::Ident, S::StringRef, S::Span>,
}

impl<S: Session> MacroCallActions<S> {
    pub fn new(parent: SemanticActions<S>, name: (S::MacroEntry, S::Span)) -> MacroCallActions<S> {
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
    {S: Session}, MacroCallActions<S>, {parent}, SemanticActions<S>, S::Span
}

impl<S: Session> MacroCallContext<S::Span> for MacroCallActions<S> {
    type Token = SemanticToken<S::Ident, S::StringRef>;
    type MacroArgContext = MacroArgContext<S>;

    fn enter_macro_arg(self) -> Self::MacroArgContext {
        MacroArgContext::new(self)
    }
}

impl<S: Session> InstrEndContext<S::Span> for MacroCallActions<S> {
    type ParentContext = SemanticActions<S>;

    fn did_parse_instr(self) -> Self::ParentContext {
        let Self {
            mut parent,
            name,
            args,
        } = self;
        parent.with_session(|session| (session.call_macro(name, args), ()));
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

    use crate::analysis::resolve::ResolvedIdent;
    use crate::analysis::semantics::tests::*;
    use crate::analysis::session::{MockMacroId, SessionEvent};
    use crate::analysis::syntax::{InstrContext, LineEndContext, Token, TokenStreamContext};

    #[test]
    fn call_nullary_macro() {
        let name = "my_macro";
        let macro_id = MockMacroId(0);
        let log = log_with_predefined_names(
            vec![(name.into(), ResolvedIdent::Macro(macro_id))],
            |actions| {
                actions
                    .will_parse_line()
                    .into_instr_line()
                    .will_parse_instr(name.into(), ())
                    .into_macro_instr()
                    .did_parse_instr()
                    .did_parse_line(())
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
        let arg_token = Token::Ident("A".into());
        let macro_id = MockMacroId(0);
        let log = log_with_predefined_names(
            vec![(name.into(), ResolvedIdent::Macro(macro_id))],
            |actions| {
                let mut call = actions
                    .will_parse_line()
                    .into_instr_line()
                    .will_parse_instr(name.into(), ())
                    .into_macro_instr();
                call = {
                    let mut arg = call.enter_macro_arg();
                    arg.push_token((arg_token.clone(), ()));
                    arg.exit()
                };
                call.did_parse_instr().did_parse_line(())
            },
        );
        assert_eq!(
            log,
            [SessionEvent::InvokeMacro(macro_id, vec![vec![arg_token]]).into()]
        )
    }
}
