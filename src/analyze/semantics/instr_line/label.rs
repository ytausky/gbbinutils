use super::{InstrLineSemantics, InstrLineState, Session};

use crate::analyze::semantics::Params;
use crate::analyze::session::ReentrancyActions;
use crate::analyze::syntax::actions::LabelActions;

pub(super) type LabelSemantics<R, N, B> = Session<R, N, B, LabelState<R>>;

pub(in crate::analyze) struct LabelState<S: ReentrancyActions> {
    parent: InstrLineState<S>,
    label: (S::Ident, S::Span),
    params: Params<S::Ident, S::Span>,
}

impl<S: ReentrancyActions> LabelState<S> {
    pub fn new(parent: InstrLineState<S>, label: (S::Ident, S::Span)) -> Self {
        Self {
            parent,
            label,
            params: (Vec::new(), Vec::new()),
        }
    }
}

impl<R: ReentrancyActions, N, B> LabelActions<R::Ident, R::Span> for LabelSemantics<R, N, B> {
    type Next = InstrLineSemantics<R, N, B>;

    fn act_on_param(&mut self, ident: R::Ident, span: R::Span) {
        let params = &mut self.state.params;
        params.0.push(ident);
        params.1.push(span)
    }

    fn did_parse_label(mut self) -> Self::Next {
        self.state.parent.label = Some((self.state.label, self.state.params));
        set_state!(self, self.state.parent)
    }
}
