use super::{InstrLineState, Keyword, Session, TokenStreamSemantics};

use crate::analyze::reentrancy::{MacroArgs, ReentrancyActions};
use crate::analyze::semantics::actions::token_line::TokenContext;
use crate::analyze::semantics::actions::TokenStreamState;
use crate::analyze::semantics::builtin_instr::{BuiltinInstr, BuiltinInstrSet, Dispatch};
use crate::analyze::semantics::resolve::{NameTable, StartScope};
use crate::analyze::semantics::TokenLineContext;
use crate::analyze::syntax::actions::{InstrFinalizer, MacroArgActions, MacroInstrActions};
use crate::analyze::{SemanticToken, TokenSeq};
use crate::object::builder::Backend;

use std::ops::DerefMut;

pub(super) type MacroInstrSemantics<I, R, N, B> = Session<I, R, N, B, MacroInstrState<R>>;

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

impl<I, R, N, B> MacroInstrActions<R::Span> for MacroInstrSemantics<I, R, N, B>
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
    BuiltinInstr<&'static I::Binding, &'static I::Free, R>: Dispatch<I, R>,
    TokenLineContext<R::Ident, R::StringRef, R::Span>: TokenContext<I, R>,
{
    type Token = SemanticToken<R::Ident, R::StringRef>;
    type MacroArgActions = MacroArgSemantics<I, R, N, B>;

    fn will_parse_macro_arg(self) -> Self::MacroArgActions {
        set_state!(self, MacroArgState::new(self.state))
    }
}

impl<I, R, N, B> InstrFinalizer<R::Span> for MacroInstrSemantics<I, R, N, B>
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
    BuiltinInstr<&'static I::Binding, &'static I::Free, R>: Dispatch<I, R>,
    TokenLineContext<R::Ident, R::StringRef, R::Span>: TokenContext<I, R>,
{
    type Next = TokenStreamSemantics<I, R, N, B>;

    fn did_parse_instr(self) -> Self::Next {
        self.reentrancy.call_macro(
            self.state.name,
            self.state.args,
            Session {
                instr_set: self.instr_set,
                reentrancy: (),
                names: self.names,
                builder: self.builder,
                state: TokenStreamState::from(self.state.parent),
                stack: self.stack,
            },
        )
    }
}

type MacroArgSemantics<I, R, N, B> = Session<I, R, N, B, MacroArgState<R>>;

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

impl<I, R: ReentrancyActions, N, B> MacroArgActions<R::Span> for MacroArgSemantics<I, R, N, B> {
    type Token = SemanticToken<R::Ident, R::StringRef>;
    type Next = MacroInstrSemantics<I, R, N, B>;

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
