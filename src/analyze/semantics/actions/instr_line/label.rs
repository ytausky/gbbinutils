use super::*;

use crate::analyze::semantics::Params;
use crate::syntax::actions::LabelContext;

pub(crate) type LabelSemantics<'a, S> = Semantics<
    'a,
    S,
    LabelState<S>,
    <S as IdentSource>::Ident,
    <S as StringSource>::StringRef,
    <S as SpanSource>::Span,
>;

pub(crate) struct LabelState<S: Session> {
    parent: InstrLineState<S::Ident, S::Span>,
    label: (S::Ident, S::Span),
    params: Params<S::Ident, S::Span>,
}

impl<S: Session> LabelState<S> {
    pub fn new(parent: InstrLineState<S::Ident, S::Span>, label: (S::Ident, S::Span)) -> Self {
        Self {
            parent,
            label,
            params: (Vec::new(), Vec::new()),
        }
    }
}

impl<'a, S: Session> LabelContext for LabelSemantics<'a, S> {
    type Next = InstrLineSemantics<'a, S>;

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
