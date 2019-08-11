use super::{InstrLineSemantics, InstrLineState, SemanticActions};

use crate::analysis::semantics::Params;
use crate::analysis::session::Session;
use crate::analysis::syntax::actions::LabelActions;

pub(super) type LabelSemantics<S> = SemanticActions<LabelState<S>, S>;

pub(in crate::analysis) struct LabelState<S: Session> {
    parent: InstrLineState<S>,
    label: (S::Ident, S::Span),
    params: Params<S::Ident, S::Span>,
}

impl<S: Session> LabelState<S> {
    pub fn new(parent: InstrLineState<S>, label: (S::Ident, S::Span)) -> Self {
        Self {
            parent,
            label,
            params: (Vec::new(), Vec::new()),
        }
    }
}

impl<S: Session> LabelActions<S::Ident, S::Span> for LabelSemantics<S> {
    type Next = InstrLineSemantics<S>;

    fn act_on_param(&mut self, ident: S::Ident, span: S::Span) {
        let params = &mut self.state.params;
        params.0.push(ident);
        params.1.push(span)
    }

    fn did_parse_label(mut self) -> Self::Next {
        self.state.parent.label = Some((self.state.label, self.state.params));
        set_state!(self, self.state.parent)
    }
}
