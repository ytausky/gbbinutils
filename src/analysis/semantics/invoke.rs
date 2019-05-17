use super::StmtActions;

use crate::analysis::session::Session;
use crate::analysis::syntax::{MacroCallContext, TokenSeqContext};
use crate::analysis::{Ident, SemanticToken, TokenSeq};
use crate::diag::DelegateDiagnostics;

pub(in crate::analysis) struct MacroCallActions<S: Session> {
    parent: StmtActions<S>,
    name: (Ident<S::StringRef>, S::Span),
    args: Vec<TokenSeq<S::StringRef, S::Span>>,
}

impl<S: Session> MacroCallActions<S> {
    pub fn new(
        parent: StmtActions<S>,
        name: (Ident<S::StringRef>, S::Span),
    ) -> MacroCallActions<S> {
        MacroCallActions {
            parent,
            name,
            args: Vec::new(),
        }
    }

    fn push_arg(&mut self, arg: Vec<(SemanticToken<S::StringRef>, S::Span)>) {
        self.args.push(arg)
    }
}

impl<S: Session> DelegateDiagnostics<S::Span> for MacroCallActions<S> {
    type Delegate = S::Delegate;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.diagnostics()
    }
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
    tokens: Vec<(SemanticToken<S::StringRef>, S::Span)>,
    parent: MacroCallActions<S>,
}

impl<S: Session> MacroArgContext<S> {
    fn new(parent: MacroCallActions<S>) -> MacroArgContext<S> {
        MacroArgContext {
            tokens: Vec::new(),
            parent,
        }
    }
}

impl<S: Session> DelegateDiagnostics<S::Span> for MacroArgContext<S> {
    type Delegate = S::Delegate;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.parent.diagnostics()
    }
}

impl<S: Session> TokenSeqContext<S::Span> for MacroArgContext<S> {
    type Token = SemanticToken<S::StringRef>;
    type Parent = MacroCallActions<S>;

    fn push_token(&mut self, token: (Self::Token, S::Span)) {
        self.tokens.push(token)
    }

    fn exit(mut self) -> Self::Parent {
        self.parent.push_arg(self.tokens);
        self.parent
    }
}
