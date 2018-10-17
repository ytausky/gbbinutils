use super::{
    analyze_reloc_expr, CommandArgs, Directive, SemanticActions, SemanticAtom, SemanticExpr,
};
use backend;
use backend::{BinaryOperator, RelocAtom};
use diagnostics::{InternalDiagnostic, Message};
use expr::ExprVariant;
use frontend::session::{ChunkId, Session};
use frontend::syntax::Literal;
use instruction::RelocExpr;
use span::Span;
use std::fmt::Debug;
use std::iter;
use Width;

pub fn analyze_directive<'a, S: Session + 'a>(
    directive: (Directive, S::Span),
    args: CommandArgs<S>,
    actions: &mut SemanticActions<'a, S>,
) -> Result<(), InternalDiagnostic<S::Span>> {
    match directive.0 {
        Directive::Db => analyze_data(Width::Byte, args, actions),
        Directive::Ds => analyze_ds(directive.1, args, actions),
        Directive::Dw => analyze_data(Width::Word, args, actions),
        Directive::Include => {
            analyze_include(args, actions);
            Ok(())
        }
        Directive::Org => analyze_org(directive.1, args, actions),
    }
}

fn analyze_data<'a, S: Session + 'a>(
    width: Width,
    args: CommandArgs<S>,
    actions: &mut SemanticActions<'a, S>,
) -> Result<(), InternalDiagnostic<S::Span>> {
    for arg in args {
        let expr = analyze_reloc_expr(arg, &mut actions.expr_factory)?;
        actions.session.emit_item(backend::Item::Data(expr, width))
    }
    Ok(())
}

fn analyze_ds<'a, S: Session + 'a>(
    span: S::Span,
    args: CommandArgs<S>,
    actions: &mut SemanticActions<'a, S>,
) -> Result<(), InternalDiagnostic<S::Span>> {
    let arg = single_arg(span, args)?;
    let count = analyze_reloc_expr(arg, &mut actions.expr_factory)?;
    actions
        .session
        .set_origin(location_counter_plus_expr(count));
    Ok(())
}

fn location_counter_plus_expr<S: Clone>(expr: RelocExpr<S>) -> RelocExpr<S> {
    let span = expr.span.clone();
    RelocExpr {
        variant: ExprVariant::Binary(
            BinaryOperator::Plus,
            Box::new(RelocExpr {
                variant: ExprVariant::Atom(RelocAtom::LocationCounter),
                span: span.clone(),
            }),
            Box::new(expr),
        ),
        span,
    }
}

fn analyze_include<'a, F: Session + 'a>(
    args: CommandArgs<F>,
    actions: &mut SemanticActions<'a, F>,
) {
    actions.session.analyze_chunk(reduce_include(args));
}

fn reduce_include<I, S: Span>(mut arguments: Vec<SemanticExpr<I, S>>) -> ChunkId<I, S> {
    assert_eq!(arguments.len(), 1);
    let path = arguments.pop().unwrap();
    match path.variant {
        ExprVariant::Atom(SemanticAtom::Literal(Literal::String(path_str))) => {
            ChunkId::File((path_str, Some(path.span)))
        }
        _ => panic!(),
    }
}

fn analyze_org<'a, S: Session + 'a>(
    span: S::Span,
    args: CommandArgs<S>,
    actions: &mut SemanticActions<'a, S>,
) -> Result<(), InternalDiagnostic<S::Span>> {
    let arg = single_arg(span, args)?;
    let expr = analyze_reloc_expr(arg, &mut actions.expr_factory)?;
    actions.session.set_origin(expr);
    Ok(())
}

fn single_arg<T: Debug + PartialEq, S>(
    span: S,
    args: impl IntoIterator<Item = T>,
) -> Result<T, InternalDiagnostic<S>> {
    let mut args = args.into_iter();
    let arg = args.next().ok_or_else(|| {
        InternalDiagnostic::new(
            Message::OperandCount {
                actual: 0,
                expected: 1,
            },
            iter::empty(),
            span,
        )
    })?;
    assert_eq!(args.next(), None);
    Ok(arg)
}
