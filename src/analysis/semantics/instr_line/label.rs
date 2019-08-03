use super::InstrLineSemantics;

use crate::analysis::semantics::Params;
use crate::analysis::session::Session;
use crate::analysis::syntax::LabelActions;

pub(in crate::analysis) struct LabelSemantics<S: Session> {
    parent: InstrLineSemantics<S>,
    label: (S::Ident, S::Span),
    params: Params<S::Ident, S::Span>,
}

impl<S: Session> LabelSemantics<S> {
    pub fn new(parent: InstrLineSemantics<S>, label: (S::Ident, S::Span)) -> Self {
        Self {
            parent,
            label,
            params: (Vec::new(), Vec::new()),
        }
    }
}

delegate_diagnostics! {
    {S: Session}, LabelSemantics<S>, {parent}, S, S::Span
}

impl<S: Session> LabelActions<S::Ident, S::Span> for LabelSemantics<S> {
    type Next = InstrLineSemantics<S>;

    fn act_on_param(&mut self, (ident, span): (S::Ident, S::Span)) {
        self.params.0.push(ident);
        self.params.1.push(span)
    }

    fn did_parse_label(mut self) -> Self::Next {
        self.parent.line.label = Some((self.label, self.params));
        self.parent
    }
}
