use super::{Ident, Params, PushOp};

use crate::model::ParamId;

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
            .map(|pos| ParamId(pos));
        if let Some(id) = param {
            self.parent.push_op(id, span)
        } else {
            self.parent.push_op(ident, span)
        }
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
