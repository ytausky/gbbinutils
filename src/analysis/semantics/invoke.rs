use super::StmtActions;

use crate::analysis::session::{MacroArgs, Session};
use crate::analysis::syntax::{MacroCallContext, TokenSeqContext};
use crate::analysis::{Ident, SemanticToken, TokenSeq};

pub(in crate::analysis) struct MacroCallActions<S: Session> {
    parent: StmtActions<S>,
    name: (Ident<S::StringRef>, S::Span),
    args: MacroArgs<S::StringRef, S::Span>,
}

impl<S: Session> MacroCallActions<S> {
    pub fn new(
        parent: StmtActions<S>,
        name: (Ident<S::StringRef>, S::Span),
    ) -> MacroCallActions<S> {
        MacroCallActions {
            parent,
            name,
            args: (Vec::new(), Vec::new()),
        }
    }

    fn push_arg(&mut self, arg: TokenSeq<S::StringRef, S::Span>) {
        self.args.0.push(arg.0);
        self.args.1.push(arg.1);
    }
}

delegate_diagnostics! {
    {S: Session}, MacroCallActions<S>, {parent}, StmtActions<S>, S::Span
}

impl<S: Session> MacroCallContext<S::Span> for MacroCallActions<S> {
    type Token = SemanticToken<S::StringRef>;
    type Parent = StmtActions<S>;
    type MacroArgContext = MacroArgContext<S>;

    fn enter_macro_arg(self) -> Self::MacroArgContext {
        MacroArgContext::new(self)
    }

    fn exit(mut self) -> Self::Parent {
        self.parent.parent.session = Some(
            self.parent
                .parent
                .session
                .take()
                .unwrap()
                .call_macro(self.name, self.args),
        );
        self.parent
    }
}

pub(in crate::analysis) struct MacroArgContext<S: Session> {
    tokens: TokenSeq<S::StringRef, S::Span>,
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
    type Token = SemanticToken<S::StringRef>;
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
