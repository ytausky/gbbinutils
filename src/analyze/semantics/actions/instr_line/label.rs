use super::{InstrLineSemantics, InstrLineState, Semantics};

use crate::analyze::reentrancy::ReentrancyActions;
use crate::analyze::semantics::Params;
use crate::analyze::syntax::actions::LabelContext;

pub(super) type LabelSemantics<'a, R, N, B> = Semantics<'a, R, N, B, LabelState<R>>;

pub(in crate::analyze) struct LabelState<R: ReentrancyActions> {
    parent: InstrLineState<R::Ident, R::Span>,
    label: (R::Ident, R::Span),
    params: Params<R::Ident, R::Span>,
}

impl<R: ReentrancyActions> LabelState<R> {
    pub fn new(parent: InstrLineState<R::Ident, R::Span>, label: (R::Ident, R::Span)) -> Self {
        Self {
            parent,
            label,
            params: (Vec::new(), Vec::new()),
        }
    }
}

impl<'a, R: ReentrancyActions, N, B> LabelContext for LabelSemantics<'a, R, N, B> {
    type Next = InstrLineSemantics<'a, R, N, B>;

    fn act_on_param(&mut self, ident: R::Ident, span: R::Span) {
        let params = &mut self.core.state.params;
        params.0.push(ident);
        params.1.push(span)
    }

    fn did_parse_label(mut self) -> Self::Next {
        self.core.state.parent.label = Some((self.core.state.label, self.core.state.params));
        set_state!(self, self.core.state.parent)
    }
}
