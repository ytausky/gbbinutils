use super::{Ident, Params, PushOp};

use crate::analysis::backend::{Finish, FinishFnDef, LocationCounter};
use crate::diag::DelegateDiagnostics;
use crate::model::{BinOp, ParamId};

pub struct ParamsAdapter<'a, P, R, S> {
    parent: P,
    params: &'a Params<R, S>,
}

impl<'a, P, R, S> ParamsAdapter<'a, P, R, S> {
    pub fn new(parent: P, params: &'a Params<R, S>) -> Self {
        Self { parent, params }
    }
}

impl<'a, P, R, S> PushOp<Ident<R>, S> for ParamsAdapter<'a, P, R, S>
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
        impl<'a, P, R, S> PushOp<$t, S> for ParamsAdapter<'a, P, R, S>
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

impl<'a, P, R, S> Finish<S> for ParamsAdapter<'a, P, R, S>
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

impl<'a, P, R, S> FinishFnDef for ParamsAdapter<'a, P, R, S>
where
    P: FinishFnDef,
    S: Clone,
{
    type Return = P::Return;

    fn finish_fn_def(self) -> Self::Return {
        self.parent.finish_fn_def()
    }
}

impl<'a, P, R, S> DelegateDiagnostics<S> for ParamsAdapter<'a, P, R, S>
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

    use crate::model::{Atom, ParamId};

    type Expr<N, S> = crate::model::Expr<Atom<LocationCounter, N>, S>;

    #[test]
    fn translate_param() {
        let name: Ident<_> = "param".into();
        let builder: Expr<_, _> = Default::default();
        let params = (vec![name.clone()], vec![()]);
        let mut adapter = ParamsAdapter::new(builder, &params);
        adapter.push_op(name, ());
        let mut expected: Expr<_, _> = Default::default();
        expected.push_op(ParamId(0), ());
        assert_eq!(adapter.parent, expected)
    }

    #[test]
    fn pass_through_non_param() {
        let param: Ident<_> = "param".into();
        let builder: Expr<_, _> = Default::default();
        let params = (vec![param.clone()], vec![()]);
        let mut adapter = ParamsAdapter::new(builder, &params);
        let unrelated: Ident<_> = "ident".into();
        adapter.push_op(unrelated.clone(), ());
        let mut expected: Expr<_, _> = Default::default();
        expected.push_op(unrelated, ());
        assert_eq!(adapter.parent, expected)
    }
}
