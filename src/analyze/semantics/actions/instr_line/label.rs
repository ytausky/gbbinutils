use super::{InstrLineSemantics, InstrLineState, Session};

use crate::analyze::reentrancy::ReentrancyActions;
use crate::analyze::semantics::Params;
use crate::analyze::syntax::actions::LabelActions;

pub(super) type LabelSemantics<I, R, N, B> = Session<R, N, B, LabelState<I, R>>;

pub(in crate::analyze) struct LabelState<I, R: ReentrancyActions> {
    parent: InstrLineState<I, R>,
    label: (R::Ident, R::Span),
    params: Params<R::Ident, R::Span>,
}

impl<I, R: ReentrancyActions> LabelState<I, R> {
    pub fn new(parent: InstrLineState<I, R>, label: (R::Ident, R::Span)) -> Self {
        Self {
            parent,
            label,
            params: (Vec::new(), Vec::new()),
        }
    }
}

impl<I, R: ReentrancyActions, N, B> LabelActions<R::Ident, R::Span> for LabelSemantics<I, R, N, B> {
    type Next = InstrLineSemantics<I, R, N, B>;

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
