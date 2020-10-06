use super::*;

use crate::assembler::semantics::Params;
use crate::assembler::syntax::actions::LabelContext;

pub(crate) type LabelSemantics<'a, S> = Semantics<'a, S, LabelState<S>>;

pub(crate) struct LabelState<S: Analysis> {
    parent: InstrLineState<S::StringRef, S::Span>,
    label: (S::StringRef, S::Span),
    params: Params<S::StringRef, S::Span>,
}

impl<S: Analysis> LabelState<S> {
    pub fn new(
        parent: InstrLineState<S::StringRef, S::Span>,
        label: (S::StringRef, S::Span),
    ) -> Self {
        Self {
            parent,
            label,
            params: (Vec::new(), Vec::new()),
        }
    }
}

impl<'a, S: Analysis> LabelContext for LabelSemantics<'a, S> {
    type Next = InstrLineSemantics<'a, S>;

    fn act_on_param(&mut self, ident: S::StringRef, span: S::Span) {
        let params = &mut self.state.params;
        params.0.push(ident);
        params.1.push(span)
    }

    fn did_parse_label(mut self) -> Self::Next {
        self.state.parent.label = Some((self.state.label, self.state.params));
        set_state!(self, self.state.parent)
    }
}
