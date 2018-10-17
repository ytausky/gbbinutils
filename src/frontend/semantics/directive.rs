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

#[cfg(test)]
mod tests {
    use super::*;
    use frontend::semantics;
    use frontend::semantics::tests::*;
    use frontend::syntax::keyword::{Command, Operand};
    use frontend::syntax::{CommandContext, ExprActions, ExprAtom, FileContext, LineActions};
    use std::borrow::Borrow;

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let actions = unary_directive(Directive::Include, |arg| {
            arg.push_atom((ExprAtom::Literal(Literal::String(filename.to_string())), ()));
        });
        assert_eq!(
            actions,
            [TestOperation::AnalyzeChunk(ChunkId::File((
                filename.to_string(),
                Some(())
            )))]
        )
    }

    #[test]
    fn set_origin() {
        let origin = 0x3000;
        let actions = unary_directive(Directive::Org, |arg| arg.push_atom(mk_literal(origin)));
        assert_eq!(actions, [TestOperation::SetOrigin(origin.into())])
    }

    #[test]
    fn emit_byte_items() {
        test_data_items_emission(Directive::Db, mk_byte, [0x42, 0x78])
    }

    #[test]
    fn emit_word_items() {
        test_data_items_emission(Directive::Dw, mk_word, [0x4332, 0x780f])
    }

    fn mk_byte(byte: &i32) -> backend::Item<()> {
        backend::Item::Data((*byte).into(), Width::Byte)
    }

    fn mk_word(word: &i32) -> backend::Item<()> {
        backend::Item::Data((*word).into(), Width::Word)
    }

    fn test_data_items_emission(
        directive: Directive,
        mk_item: impl Fn(&i32) -> backend::Item<()>,
        data: impl Borrow<[i32]>,
    ) {
        let actions = with_directive(directive, |mut command| {
            for datum in data.borrow().iter() {
                let mut arg = command.add_argument();
                arg.push_atom(mk_literal(*datum));
                command = arg.exit();
            }
            command
        });
        assert_eq!(
            actions,
            data.borrow()
                .iter()
                .map(mk_item)
                .map(TestOperation::EmitItem)
                .collect::<Vec<_>>()
        )
    }

    #[test]
    fn reserve_3_bytes() {
        let actions = ds(|arg| arg.push_atom(mk_literal(3)));
        assert_eq!(
            actions,
            [TestOperation::SetOrigin(
                ExprVariant::Binary(
                    BinaryOperator::Plus,
                    Box::new(RelocAtom::LocationCounter.into()),
                    Box::new(3.into()),
                ).into()
            )]
        )
    }

    fn mk_literal(n: i32) -> (ExprAtom<String, Literal<String>>, ()) {
        (ExprAtom::Literal(Literal::Number(n)), ())
    }

    #[test]
    fn ds_with_malformed_expr() {
        let actions =
            ds(|arg| arg.push_atom((ExprAtom::Literal(Literal::Operand(Operand::A)), ())));
        assert_eq!(
            actions,
            [TestOperation::EmitDiagnostic(InternalDiagnostic::new(
                Message::KeywordInExpr,
                iter::once(()),
                (),
            ))]
        )
    }

    #[test]
    fn ds_without_args() {
        test_unary_directive_without_args(Directive::Ds)
    }

    #[test]
    fn org_without_args() {
        test_unary_directive_without_args(Directive::Org)
    }

    #[test]
    fn data_with_malformed_expr() {
        let actions = unary_directive(Directive::Db, |arg| {
            arg.push_atom((ExprAtom::Literal(Literal::Operand(Operand::A)), ()))
        });
        assert_eq!(
            actions,
            [TestOperation::EmitDiagnostic(InternalDiagnostic::new(
                Message::KeywordInExpr,
                iter::once(()),
                (),
            ))]
        )
    }

    fn ds(
        f: impl for<'a> FnOnce(&mut semantics::ExprActions<'a, TestFrontend>),
    ) -> Vec<TestOperation> {
        unary_directive(Directive::Ds, f)
    }

    fn unary_directive<F>(directive: Directive, f: F) -> Vec<TestOperation>
    where
        F: for<'a> FnOnce(&mut semantics::ExprActions<'a, TestFrontend>),
    {
        with_directive(directive, |command| {
            let mut arg = command.add_argument();
            f(&mut arg);
            arg.exit()
        })
    }

    fn test_unary_directive_without_args(directive: Directive) {
        let actions = with_directive(directive, |command| command);
        assert_eq!(
            actions,
            [TestOperation::EmitDiagnostic(InternalDiagnostic::new(
                Message::OperandCount {
                    actual: 0,
                    expected: 1
                },
                iter::empty(),
                (),
            ))]
        )
    }

    fn with_directive<F>(directive: Directive, f: F) -> Vec<TestOperation>
    where
        F: for<'a> FnOnce(semantics::CommandActions<'a, TestFrontend>)
            -> semantics::CommandActions<'a, TestFrontend>,
    {
        collect_semantic_actions(|actions| {
            let command = actions
                .enter_line(None)
                .enter_command((Command::Directive(directive), ()));
            f(command).exit().exit()
        })
    }
}
