use super::{InstrLineState, Keyword, Semantics, TokenStreamSemantics};

use crate::analyze::semantics::actions::TokenStreamState;
use crate::analyze::semantics::reentrancy::{MacroArgs, Meta, ReentrancyActions};
use crate::analyze::semantics::resolve::{NameTable, StartScope};
use crate::analyze::semantics::Core;
use crate::analyze::syntax::actions::{InstrFinalizer, MacroArgContext, MacroInstrContext};
use crate::analyze::{SemanticToken, TokenSeq};
use crate::object::builder::Backend;

use std::ops::DerefMut;

pub(super) type MacroInstrSemantics<'a, R, N, B> = Semantics<'a, R, N, B, MacroInstrState<R>>;

pub(in crate::analyze) struct MacroInstrState<R: Meta> {
    parent: InstrLineState<R::Ident, R::Span>,
    name: (R::MacroId, R::Span),
    args: MacroArgs<R::Ident, R::StringRef, R::Span>,
}

impl<R: Meta> MacroInstrState<R> {
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

impl<'a, R, N, B> MacroInstrContext for MacroInstrSemantics<'a, R, N, B>
where
    R: Meta,
    R::Ident: 'static,
    R::StringRef: 'static,
    R::Span: 'static,
    Core<R, N, B>: ReentrancyActions<
        Ident = R::Ident,
        StringRef = R::StringRef,
        Span = R::Span,
        MacroId = R::MacroId,
    >,
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
    type MacroArgContext = MacroArgSemantics<'a, R, N, B>;

    fn will_parse_macro_arg(self) -> Self::MacroArgContext {
        set_state!(self, MacroArgState::new(self.state))
    }
}

impl<'a, R, N, B> InstrFinalizer for MacroInstrSemantics<'a, R, N, B>
where
    R: Meta,
    R::Ident: 'static,
    R::StringRef: 'static,
    R::Span: 'static,
    Core<R, N, B>: ReentrancyActions<
        Ident = R::Ident,
        StringRef = R::StringRef,
        Span = R::Span,
        MacroId = R::MacroId,
    >,
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
        let core = self.core.call_macro(self.state.name, self.state.args);
        Semantics {
            core,
            state: TokenStreamState::from(self.state.parent),
            tokens: self.tokens,
        }
    }
}

type MacroArgSemantics<'a, R, N, B> = Semantics<'a, R, N, B, MacroArgState<R>>;

pub(in crate::analyze) struct MacroArgState<R: Meta> {
    tokens: TokenSeq<R::Ident, R::StringRef, R::Span>,
    parent: MacroInstrState<R>,
}

impl<R: Meta> MacroArgState<R> {
    fn new(parent: MacroInstrState<R>) -> Self {
        Self {
            tokens: (Vec::new(), Vec::new()),
            parent,
        }
    }
}

impl<'a, R: Meta, N, B> MacroArgContext for MacroArgSemantics<'a, R, N, B> {
    type Next = MacroInstrSemantics<'a, R, N, B>;

    fn act_on_token(&mut self, token: (SemanticToken<R::Ident, R::StringRef>, R::Span)) {
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
    use crate::analyze::semantics::actions::tests::*;
    use crate::analyze::semantics::reentrancy::ReentrancyEvent;
    use crate::analyze::semantics::resolve::ResolvedName;
    use crate::analyze::syntax::actions::{InstrContext, LineFinalizer, TokenStreamContext};
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
