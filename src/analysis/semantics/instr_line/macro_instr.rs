use super::{InstrLineState, SemanticActions, TokenStreamSemantics};

use crate::analysis::semantics::TokenStreamState;
use crate::analysis::session::{MacroArgs, Session};
use crate::analysis::syntax::{InstrFinalizer, MacroArgActions, MacroInstrActions};
use crate::analysis::{SemanticToken, TokenSeq};

pub(super) type MacroInstrSemantics<S> = SemanticActions<MacroInstrState<S>, S>;

pub(in crate::analysis) struct MacroInstrState<S: Session> {
    parent: InstrLineState<S>,
    name: (S::MacroEntry, S::Span),
    args: MacroArgs<S::Ident, S::StringRef, S::Span>,
}

impl<S: Session> MacroInstrState<S> {
    pub fn new(parent: InstrLineState<S>, name: (S::MacroEntry, S::Span)) -> Self {
        Self {
            parent,
            name,
            args: (Vec::new(), Vec::new()),
        }
    }

    fn push_arg(&mut self, arg: TokenSeq<S::Ident, S::StringRef, S::Span>) {
        let args = &mut self.args;
        args.0.push(arg.0);
        args.1.push(arg.1);
    }
}

impl<S: Session> MacroInstrActions<S::Span> for MacroInstrSemantics<S> {
    type Token = SemanticToken<S::Ident, S::StringRef>;
    type MacroArgActions = MacroArgSemantics<S>;

    fn will_parse_macro_arg(self) -> Self::MacroArgActions {
        set_state!(self, MacroArgState::new(self.state))
    }
}

impl<S: Session> InstrFinalizer<S::Span> for MacroInstrSemantics<S> {
    type Next = TokenStreamSemantics<S>;

    fn did_parse_instr(self) -> Self::Next {
        self.session.call_macro(
            self.state.name,
            self.state.args,
            TokenStreamState::from(self.state.parent),
        )
    }
}

type MacroArgSemantics<S> = SemanticActions<MacroArgState<S>, S>;

pub(in crate::analysis) struct MacroArgState<S: Session> {
    tokens: TokenSeq<S::Ident, S::StringRef, S::Span>,
    parent: MacroInstrState<S>,
}

impl<S: Session> MacroArgState<S> {
    fn new(parent: MacroInstrState<S>) -> Self {
        Self {
            tokens: (Vec::new(), Vec::new()),
            parent,
        }
    }
}

impl<S: Session> MacroArgActions<S::Span> for MacroArgSemantics<S> {
    type Token = SemanticToken<S::Ident, S::StringRef>;
    type Next = MacroInstrSemantics<S>;

    fn act_on_token(&mut self, token: (Self::Token, S::Span)) {
        let tokens = &mut self.state.tokens;
        tokens.0.push(token.0);
        tokens.1.push(token.1);
    }

    fn did_parse_macro_arg(mut self) -> Self::Next {
        self.state.parent.push_arg(self.state.tokens);
        set_state!(self, self.state.parent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::resolve::ResolvedIdent;
    use crate::analysis::semantics::tests::*;
    use crate::analysis::session::{MockMacroId, SessionEvent};
    use crate::analysis::syntax::{InstrActions, LineFinalizer, Token, TokenStreamActions};

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
                    .act_on_eos(())
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
                    let mut arg = call.will_parse_macro_arg();
                    arg.act_on_token((arg_token.clone(), ()));
                    arg.did_parse_macro_arg()
                };
                call.did_parse_instr().did_parse_line(()).act_on_eos(())
            },
        );
        assert_eq!(
            log,
            [SessionEvent::InvokeMacro(macro_id, vec![vec![arg_token]]).into()]
        )
    }
}
