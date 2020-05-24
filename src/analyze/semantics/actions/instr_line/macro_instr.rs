use super::{Core, InstrLineState, Keyword, Session, TokenStreamSemantics};

use crate::analyze::reentrancy::{MacroArgs, ReentrancyActions};
use crate::analyze::semantics::actions::TokenStreamState;
use crate::analyze::semantics::resolve::{NameTable, StartScope};
use crate::analyze::syntax::actions::{InstrFinalizer, MacroArgActions, MacroInstrActions};
use crate::analyze::{SemanticToken, TokenSeq};
use crate::object::builder::Backend;

use std::ops::DerefMut;

pub(super) type MacroInstrSemantics<'a, R, N, B> = Session<'a, R, N, B, MacroInstrState<R>>;

pub(in crate::analyze) struct MacroInstrState<R: ReentrancyActions> {
    parent: InstrLineState<R::Ident, R::Span>,
    name: (R::MacroId, R::Span),
    args: MacroArgs<R::Ident, R::StringRef, R::Span>,
}

impl<R: ReentrancyActions> MacroInstrState<R> {
    pub fn new(parent: InstrLineState<R::Ident, R::Span>, name: (R::MacroId, R::Span)) -> Self {
        Self {
            parent,
            name,
            args: (Vec::new(), Vec::new()),
        }
    }

    fn push_arg(&mut self, arg: TokenSeq<R::Ident, R::StringRef, R::Span>) {
        let args = &mut self.args;
        args.0.push(arg.0);
        args.1.push(arg.1);
    }
}

impl<'a, R, N, B> MacroInstrActions for MacroInstrSemantics<'a, R, N, B>
where
    R: ReentrancyActions,
    R::Ident: 'static,
    R::StringRef: 'static,
    R::Span: 'static,
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
    type MacroArgActions = MacroArgSemantics<'a, R, N, B>;

    fn will_parse_macro_arg(self) -> Self::MacroArgActions {
        set_state!(self, MacroArgState::new(self.core.state))
    }
}

impl<'a, R, N, B> InstrFinalizer for MacroInstrSemantics<'a, R, N, B>
where
    R: ReentrancyActions,
    R::Ident: 'static,
    R::StringRef: 'static,
    R::Span: 'static,
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
    type Next = TokenStreamSemantics<'a, R, N, B>;

    fn did_parse_instr(self) -> Self::Next {
        let (reentrancy, core) = self.reentrancy.call_macro(
            self.core.state.name,
            self.core.state.args,
            Core {
                names: self.core.names,
                builder: self.core.builder,
                state: TokenStreamState::from(self.core.state.parent),
            },
        );
        Session {
            reentrancy,
            core,
            tokens: self.tokens,
        }
    }
}

type MacroArgSemantics<'a, R, N, B> = Session<'a, R, N, B, MacroArgState<R>>;

pub(in crate::analyze) struct MacroArgState<R: ReentrancyActions> {
    tokens: TokenSeq<R::Ident, R::StringRef, R::Span>,
    parent: MacroInstrState<R>,
}

impl<R: ReentrancyActions> MacroArgState<R> {
    fn new(parent: MacroInstrState<R>) -> Self {
        Self {
            tokens: (Vec::new(), Vec::new()),
            parent,
        }
    }
}

impl<'a, R: ReentrancyActions, N, B> MacroArgActions for MacroArgSemantics<'a, R, N, B> {
    type Next = MacroInstrSemantics<'a, R, N, B>;

    fn act_on_token(&mut self, token: (SemanticToken<R::Ident, R::StringRef>, R::Span)) {
        let tokens = &mut self.core.state.tokens;
        tokens.0.push(token.0);
        tokens.1.push(token.1);
    }

    fn did_parse_macro_arg(mut self) -> Self::Next {
        self.core.state.parent.push_arg(self.core.state.tokens);
        set_state!(self, self.core.state.parent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analyze::macros::mock::MockMacroId;
    use crate::analyze::reentrancy::ReentrancyEvent;
    use crate::analyze::semantics::actions::tests::*;
    use crate::analyze::semantics::resolve::ResolvedName;
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
