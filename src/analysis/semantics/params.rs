use super::{Ident, Params, PushOp};

use crate::analysis::backend::LocationCounter;
use crate::analysis::session::Finish;
use crate::diag::DelegateDiagnostics;
use crate::model::{BinOp, ParamId};

pub struct ParamsAdapter<P, R, S> {
    parent: P,
    params: Params<R, S>,
}

impl<P, R, S> ParamsAdapter<P, R, S> {
    pub fn new(parent: P, params: Params<R, S>) -> Self {
        Self { parent, params }
    }
}

impl<P, R, S> PushOp<Ident<R>, S> for ParamsAdapter<P, R, S>
where
    P: PushOp<Ident<R>, S> + PushOp<ParamId, S>,
    R: Eq,
    S: Clone,
{
    fn push_op(&mut self, ident: Ident<R>, span: S) {
        let param = self
            .params
            .0
            .iter()
            .position(|param| param.name == ident.name)
            .map(ParamId);
        if let Some(id) = param {
            self.parent.push_op(id, span)
        } else {
            self.parent.push_op(ident, span)
        }
    }
}

macro_rules! impl_push_op_for_params_adapter {
    ($t:ty) => {
        impl<P, R, S> PushOp<$t, S> for ParamsAdapter<P, R, S>
        where
            P: PushOp<$t, S>,
            S: Clone,
        {
            fn push_op(&mut self, op: $t, span: S) {
                self.parent.push_op(op, span)
            }
        }
    };
}

impl_push_op_for_params_adapter! {LocationCounter}
impl_push_op_for_params_adapter! {i32}
impl_push_op_for_params_adapter! {BinOp}

impl<P, R, S> Finish<S> for ParamsAdapter<P, R, S>
where
    P: Finish<S>,
    S: Clone,
{
    type Parent = P::Parent;
    type Value = P::Value;

    fn finish(self) -> (Self::Parent, Self::Value) {
        self.parent.finish()
    }
}

impl<P, R, S> DelegateDiagnostics<S> for ParamsAdapter<P, R, S>
where
    P: DelegateDiagnostics<S>,
    S: Clone,
{
    type Delegate = P::Delegate;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.diagnostics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::model::{Expr, ParamId};

    #[test]
    fn translate_param() {
        let name: Ident<_> = "param".into();
        let builder: Expr<_, _> = Default::default();
        let mut adapter = ParamsAdapter::new(builder, (vec![name.clone()], vec![()]));
        adapter.push_op(name, ());
        let mut expected: Expr<_, _> = Default::default();
        expected.push_op(ParamId(0), ());
        assert_eq!(adapter.parent, expected)
    }

    #[test]
    fn pass_through_non_param() {
        let param: Ident<_> = "param".into();
        let builder: Expr<_, _> = Default::default();
        let mut adapter = ParamsAdapter::new(builder, (vec![param.clone()], vec![()]));
        let unrelated: Ident<_> = "ident".into();
        adapter.push_op(unrelated.clone(), ());
        let mut expected: Expr<_, _> = Default::default();
        expected.push_op(unrelated, ());
        assert_eq!(adapter.parent, expected)
    }
}
