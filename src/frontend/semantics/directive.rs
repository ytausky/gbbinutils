use super::{
    analyze_reloc_expr, CommandArgs, Directive, SemanticActions, SemanticAtom, SemanticExpr,
};
use crate::backend;
use crate::backend::{Backend, BinaryOperator, ValueBuilder, Width};
use crate::diagnostics::{CompactDiagnostic, Diagnostics, Message};
use crate::expr::ExprVariant;
use crate::frontend::syntax::Literal;
use crate::frontend::Frontend;
use crate::span::Source;
use std::fmt::Debug;

pub fn analyze_directive<'a: 'b, 'b, F: Frontend<D>, B: Backend<D::Span>, D: Diagnostics>(
    directive: (Directive, D::Span),
    args: CommandArgs<F::Ident, D::Span>,
    actions: &'b mut SemanticActions<'a, F, B, D>,
) -> Result<(), CompactDiagnostic<D::Span>> {
    let context = DirectiveContext {
        span: directive.1,
        args,
        actions,
    };
    context.analyze(directive.0)
}

struct DirectiveContext<'a, A, I, S> {
    span: S,
    args: CommandArgs<I, S>,
    actions: &'a mut A,
}

impl<'a, 'b, F, B, D> DirectiveContext<'b, SemanticActions<'a, F, B, D>, F::Ident, D::Span>
where
    'a: 'b,
    F: Frontend<D>,
    B: Backend<D::Span>,
    D: Diagnostics,
{
    fn analyze(self, directive: Directive) -> Result<(), CompactDiagnostic<D::Span>> {
        match directive {
            Directive::Db => self.analyze_data(Width::Byte),
            Directive::Ds => self.analyze_ds(),
            Directive::Dw => self.analyze_data(Width::Word),
            Directive::Equ => self.analyze_equ(),
            Directive::Include => self.analyze_include(),
            Directive::Org => self.analyze_org(),
        }
    }

    fn analyze_data(self, width: Width) -> Result<(), CompactDiagnostic<D::Span>> {
        for arg in self.args {
            let expr = analyze_reloc_expr(arg, &mut self.actions.session.backend.build_value())?;
            self.actions
                .session
                .backend
                .emit_item(backend::Item::Data(expr, width))
        }
        Ok(())
    }

    fn analyze_ds(self) -> Result<(), CompactDiagnostic<D::Span>> {
        let origin = {
            let arg = single_arg(self.span, self.args)?;
            let builder = &mut self.actions.session.backend.build_value();
            let count = analyze_reloc_expr(arg, builder)?;
            location_counter_plus_expr(count, builder)
        };
        self.actions.session.backend.set_origin(origin);
        Ok(())
    }

    fn analyze_equ(self) -> Result<(), CompactDiagnostic<D::Span>> {
        let symbol = self.actions.label.take().unwrap();
        let arg = single_arg(self.span, self.args)?;
        let value = analyze_reloc_expr(arg, &mut self.actions.session.backend.build_value())?;
        self.actions.session.backend.define_symbol(symbol, value);
        Ok(())
    }

    fn analyze_include(self) -> Result<(), CompactDiagnostic<D::Span>> {
        let (path, span) = reduce_include(self.span, self.args)?;
        self.actions
            .session
            .analyze_file(path)
            .map_err(|err| CompactDiagnostic::new(err.into(), span))
    }

    fn analyze_org(self) -> Result<(), CompactDiagnostic<D::Span>> {
        let arg = single_arg(self.span, self.args)?;
        let expr = analyze_reloc_expr(arg, &mut self.actions.session.backend.build_value())?;
        self.actions.session.backend.set_origin(expr);
        Ok(())
    }
}

fn location_counter_plus_expr<V: Source, B: ValueBuilder<V>>(expr: V, builder: &mut B) -> V {
    let location = builder.location(expr.span());
    builder.apply_binary_operator((BinaryOperator::Plus, expr.span()), location, expr)
}

fn reduce_include<I, S>(
    span: S,
    args: Vec<SemanticExpr<I, S>>,
) -> Result<(I, S), CompactDiagnostic<S>>
where
    I: Debug + PartialEq,
    S: Debug + PartialEq,
{
    let arg = single_arg(span, args)?;
    match arg.variant {
        ExprVariant::Atom(SemanticAtom::Literal(Literal::String(path))) => Ok((path, arg.span)),
        _ => Err(CompactDiagnostic::new(Message::ExpectedString, arg.span)),
    }
}

fn single_arg<T: Debug + PartialEq, S>(
    span: S,
    args: impl IntoIterator<Item = T>,
) -> Result<T, CompactDiagnostic<S>> {
    let mut args = args.into_iter();
    let arg = args.next().ok_or_else(|| {
        CompactDiagnostic::new(
            Message::OperandCount {
                actual: 0,
                expected: 1,
            },
            span,
        )
    })?;
    assert_eq!(args.next(), None);
    Ok(arg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{RelocAtom, RelocExpr};
    use crate::codebase::CodebaseError;
    use crate::frontend::semantics;
    use crate::frontend::semantics::tests::*;
    use crate::frontend::session::Session;
    use crate::frontend::syntax::keyword::{Command, Operand};
    use crate::frontend::syntax::{
        CommandContext, ExprAtom, ExprContext, FileContext, StmtContext,
    };
    use std::borrow::Borrow;
    use std::cell::RefCell;
    use std::io;

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let actions = unary_directive(Directive::Include, |arg| {
            arg.push_atom((ExprAtom::Literal(Literal::String(filename.to_string())), ()));
        });
        assert_eq!(actions, [TestOperation::AnalyzeFile(filename.to_string())])
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

    fn mk_byte(byte: &i32) -> backend::Item<RelocExpr<()>> {
        backend::Item::Data((*byte).into(), Width::Byte)
    }

    fn mk_word(word: &i32) -> backend::Item<RelocExpr<()>> {
        backend::Item::Data((*word).into(), Width::Word)
    }

    fn test_data_items_emission(
        directive: Directive,
        mk_item: impl Fn(&i32) -> backend::Item<RelocExpr<()>>,
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
                )
                .into()
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
            [TestOperation::EmitDiagnostic(CompactDiagnostic::new(
                Message::KeywordInExpr { keyword: () },
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
    fn include_without_args() {
        test_unary_directive_without_args(Directive::Include)
    }

    #[test]
    fn include_with_number() {
        let actions = unary_directive(Directive::Include, |arg| arg.push_atom(mk_literal(7)));
        assert_eq!(
            actions,
            [TestOperation::EmitDiagnostic(CompactDiagnostic::new(
                Message::ExpectedString,
                (),
            ))]
        )
    }

    #[test]
    fn data_with_malformed_expr() {
        let actions = unary_directive(Directive::Db, |arg| {
            arg.push_atom((ExprAtom::Literal(Literal::Operand(Operand::A)), ()))
        });
        assert_eq!(
            actions,
            [TestOperation::EmitDiagnostic(CompactDiagnostic::new(
                Message::KeywordInExpr { keyword: () },
                (),
            ))]
        )
    }

    #[test]
    fn include_file_with_invalid_utf8() {
        let name = "invalid_utf8.s";
        let operations = RefCell::new(Vec::new());
        let mut frontend = TestFrontend::new(&operations);
        frontend.fail(CodebaseError::Utf8Error);
        let mut backend = TestBackend::new(&operations);
        let mut diagnostics = TestDiagnostics::new(&operations);
        let session = Session::new(&mut frontend, &mut backend, &mut diagnostics);
        {
            let mut context = SemanticActions::new(session)
                .enter_stmt(None)
                .enter_command((Command::Directive(Directive::Include), ()))
                .add_argument();
            context.push_atom((ExprAtom::Literal(Literal::String(name.into())), ()));
            context.exit().exit().exit();
        }
        assert_eq!(
            operations.into_inner(),
            [
                TestOperation::AnalyzeFile(name.into()),
                TestOperation::EmitDiagnostic(CompactDiagnostic::new(Message::InvalidUtf8, ()))
            ]
        )
    }

    #[test]
    fn include_nonexistent_file() {
        let name = "nonexistent.s";
        let message = "some message";
        let operations = RefCell::new(Vec::new());
        let mut frontend = TestFrontend::new(&operations);
        frontend.fail(CodebaseError::IoError(io::Error::new(
            io::ErrorKind::NotFound,
            message,
        )));
        let mut backend = TestBackend::new(&operations);
        let mut diagnostics = TestDiagnostics::new(&operations);
        let session = Session::new(&mut frontend, &mut backend, &mut diagnostics);
        {
            let mut context = SemanticActions::new(session)
                .enter_stmt(None)
                .enter_command((Command::Directive(Directive::Include), ()))
                .add_argument();
            context.push_atom((ExprAtom::Literal(Literal::String(name.into())), ()));
            context.exit().exit().exit();
        }
        assert_eq!(
            operations.into_inner(),
            [
                TestOperation::AnalyzeFile(name.into()),
                TestOperation::EmitDiagnostic(CompactDiagnostic::new(
                    Message::IoError {
                        string: message.to_string()
                    },
                    ()
                ))
            ]
        )
    }

    #[test]
    fn define_symbol() {
        let symbol = "sym";
        let value = 3;
        let actions = with_labeled_directive(symbol, Directive::Equ, |arg| {
            arg.push_atom((ExprAtom::Literal(Literal::Number(value)), ()))
        });
        assert_eq!(
            actions,
            [TestOperation::DefineSymbol(symbol.into(), value.into())]
        )
    }

    fn ds(
        f: impl for<'a> FnOnce(
            &mut semantics::ExprContext<'a, TestFrontend<'a>, TestBackend<'a>, TestDiagnostics<'a>>,
        ),
    ) -> Vec<TestOperation> {
        unary_directive(Directive::Ds, f)
    }

    type TestExprContext<'a> =
        semantics::ExprContext<'a, TestFrontend<'a>, TestBackend<'a>, TestDiagnostics<'a>>;

    fn unary_directive<F>(directive: Directive, f: F) -> Vec<TestOperation>
    where
        F: for<'a> FnOnce(&mut TestExprContext<'a>),
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
            [TestOperation::EmitDiagnostic(CompactDiagnostic::new(
                Message::OperandCount {
                    actual: 0,
                    expected: 1
                },
                (),
            ))]
        )
    }

    type TestCommandActions<'a> =
        semantics::CommandActions<'a, TestFrontend<'a>, TestBackend<'a>, TestDiagnostics<'a>>;

    fn with_directive<F>(directive: Directive, f: F) -> Vec<TestOperation>
    where
        F: for<'a> FnOnce(TestCommandActions<'a>) -> TestCommandActions<'a>,
    {
        collect_semantic_actions(|actions| {
            let command = actions
                .enter_stmt(None)
                .enter_command((Command::Directive(directive), ()));
            f(command).exit().exit()
        })
    }

    fn with_labeled_directive<F>(label: &str, directive: Directive, f: F) -> Vec<TestOperation>
    where
        F: for<'a> FnOnce(&mut TestExprContext<'a>),
    {
        collect_semantic_actions(|actions| {
            let mut arg = actions
                .enter_stmt(Some((label.into(), ())))
                .enter_command((Command::Directive(directive), ()))
                .add_argument();
            f(&mut arg);
            arg.exit().exit().exit()
        })
    }
}
