use super::{InstrLineState, Keyword, Session, TokenStreamSemantics};

use crate::analyze::reentrancy::{MacroArgs, ReentrancyActions};
use crate::analyze::resolve::{NameTable, StartScope};
use crate::analyze::semantics::TokenStreamState;
use crate::analyze::syntax::actions::{InstrFinalizer, MacroArgActions, MacroInstrActions};
use crate::analyze::{SemanticToken, TokenSeq};
use crate::object::builder::Backend;

use std::ops::DerefMut;

pub(super) type MacroInstrSemantics<R, N, B> = Session<R, N, B, MacroInstrState<R>>;

pub(in crate::analyze) struct MacroInstrState<S: ReentrancyActions> {
    parent: InstrLineState<S>,
    name: (S::MacroId, S::Span),
    args: MacroArgs<S::Ident, S::StringRef, S::Span>,
}

impl<S: ReentrancyActions> MacroInstrState<S> {
    pub fn new(parent: InstrLineState<S>, name: (S::MacroId, S::Span)) -> Self {
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

impl<R, N, B> MacroInstrActions<R::Span> for MacroInstrSemantics<R, N, B>
where
    R: ReentrancyActions,
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
    type Token = SemanticToken<R::Ident, R::StringRef>;
    type MacroArgActions = MacroArgSemantics<R, N, B>;

    fn will_parse_macro_arg(self) -> Self::MacroArgActions {
        set_state!(self, MacroArgState::new(self.state))
    }
}

impl<R, N, B> InstrFinalizer<R::Span> for MacroInstrSemantics<R, N, B>
where
    R: ReentrancyActions,
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
    type Next = TokenStreamSemantics<R, N, B>;

    fn did_parse_instr(self) -> Self::Next {
        self.reentrancy.call_macro(
            self.state.name,
            self.state.args,
            Session {
                reentrancy: (),
                names: self.names,
                builder: self.builder,
                state: TokenStreamState::from(self.state.parent),
            },
        )
    }
}

type MacroArgSemantics<R, N, B> = Session<R, N, B, MacroArgState<R>>;

pub(in crate::analyze) struct MacroArgState<S: ReentrancyActions> {
    tokens: TokenSeq<S::Ident, S::StringRef, S::Span>,
    parent: MacroInstrState<S>,
}

impl<S: ReentrancyActions> MacroArgState<S> {
    fn new(parent: MacroInstrState<S>) -> Self {
        Self {
            tokens: (Vec::new(), Vec::new()),
            parent,
        }
    }
}

impl<R: ReentrancyActions, N, B> MacroArgActions<R::Span> for MacroArgSemantics<R, N, B> {
    type Token = SemanticToken<R::Ident, R::StringRef>;
    type Next = MacroInstrSemantics<R, N, B>;

    fn act_on_token(&mut self, token: (Self::Token, R::Span)) {
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

    use crate::analyze::macros::mock::MockMacroId;
    use crate::analyze::reentrancy::ReentrancyEvent;
    use crate::analyze::resolve::ResolvedName;
    use crate::analyze::semantics::tests::*;
    use crate::analyze::syntax::actions::{InstrActions, LineFinalizer, TokenStreamActions};
    use crate::analyze::syntax::Token;

    #[test]
    fn call_nullary_macro() {
        let name = "my_macro";
        let macro_id = MockMacroId(0);
        let log = log_with_predefined_names(
            vec![(name.into(), ResolvedName::Macro(macro_id))],
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
            [ReentrancyEvent::InvokeMacro(macro_id, Vec::new()).into()]
        )
    }

    #[test]
    fn call_unary_macro() {
        let name = "my_macro";
        let arg_token = Token::Ident("A".into());
        let macro_id = MockMacroId(0);
        let log = log_with_predefined_names(
            vec![(name.into(), ResolvedName::Macro(macro_id))],
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
            [ReentrancyEvent::InvokeMacro(macro_id, vec![vec![arg_token]]).into()]
        )
    }
}
