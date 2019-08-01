use super::{InstrLineSemantics, TokenStreamSemantics};

use crate::analysis::session::{MacroArgs, Session};
use crate::analysis::syntax::{InstrFinalizer, MacroArgActions, MacroInstrActions};
use crate::analysis::{SemanticToken, TokenSeq};

pub(in crate::analysis) struct MacroInstrSemantics<S: Session> {
    parent: InstrLineSemantics<S>,
    name: (S::MacroEntry, S::Span),
    args: MacroArgs<S::Ident, S::StringRef, S::Span>,
}

impl<S: Session> MacroInstrSemantics<S> {
    pub fn new(
        parent: InstrLineSemantics<S>,
        name: (S::MacroEntry, S::Span),
    ) -> MacroInstrSemantics<S> {
        MacroInstrSemantics {
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
    {S: Session}, MacroInstrSemantics<S>, {parent}, InstrLineSemantics<S>, S::Span
}

impl<S: Session> MacroInstrActions<S::Span> for MacroInstrSemantics<S> {
    type Token = SemanticToken<S::Ident, S::StringRef>;
    type MacroArgActions = MacroArgSemantics<S>;

    fn will_parse_macro_arg(self) -> Self::MacroArgActions {
        MacroArgSemantics::new(self)
    }
}

impl<S: Session> InstrFinalizer<S::Span> for MacroInstrSemantics<S> {
    type Next = TokenStreamSemantics<S>;

    fn did_parse_instr(self) -> Self::Next {
        let Self {
            mut parent,
            name,
            args,
        } = self;
        parent.session = parent.session.call_macro(name, args);
        parent.into()
    }
}

pub(in crate::analysis) struct MacroArgSemantics<S: Session> {
    tokens: TokenSeq<S::Ident, S::StringRef, S::Span>,
    parent: MacroInstrSemantics<S>,
}

impl<S: Session> MacroArgSemantics<S> {
    fn new(parent: MacroInstrSemantics<S>) -> MacroArgSemantics<S> {
        Self {
            tokens: (Vec::new(), Vec::new()),
            parent,
        }
    }
}

delegate_diagnostics! {
    {S: Session}, MacroArgSemantics<S>, {parent}, MacroInstrSemantics<S>, S::Span
}

impl<S: Session> MacroArgActions<S::Span> for MacroArgSemantics<S> {
    type Token = SemanticToken<S::Ident, S::StringRef>;
    type Next = MacroInstrSemantics<S>;

    fn act_on_token(&mut self, token: (Self::Token, S::Span)) {
        self.tokens.0.push(token.0);
        self.tokens.1.push(token.1);
    }

    fn did_parse_macro_arg(mut self) -> Self::Next {
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
