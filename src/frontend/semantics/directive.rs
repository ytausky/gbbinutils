use super::{
    AnalyzeExpr, CommandArgs, Directive, SemanticActions, SemanticAtom, SemanticExpr, ValueContext,
};
use crate::backend;
use crate::backend::{Backend, LocationCounter, ValueBuilder, Width};
use crate::diag::*;
use crate::expr::{BinaryOperator, ExprVariant};
use crate::frontend::macros::MacroEntry;
use crate::frontend::{Frontend, Ident, Literal};
use crate::span::Source;

pub(super) fn analyze_directive<'a, 'b, F, B, D>(
    directive: (Directive, D::Span),
    args: CommandArgs<F::StringRef, D::Span>,
    actions: &'b mut SemanticActions<'a, F, B, D>,
) where
    'a: 'b,
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, MacroEntry<F, D>> + ?Sized,
    D: Diagnostics,
{
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

impl<'a, 'b, F, B, D> DelegateDiagnostics<D::Span>
    for DirectiveContext<'b, SemanticActions<'a, F, B, D>, F::StringRef, D::Span>
where
    'a: 'b,
    F: Frontend<D>,
    D: Diagnostics,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.actions.diagnostics()
    }
}

impl<'a, 'b, F, B, D> DirectiveContext<'b, SemanticActions<'a, F, B, D>, F::StringRef, D::Span>
where
    'a: 'b,
    F: Frontend<D>,
    B: Backend<Ident<F::StringRef>, D::Span, MacroEntry<F, D>> + ?Sized,
    D: Diagnostics,
{
    fn analyze(self, directive: Directive) {
        match directive {
            Directive::Db => self.analyze_data(Width::Byte),
            Directive::Ds => self.analyze_ds(),
            Directive::Dw => self.analyze_data(Width::Word),
            Directive::Equ => self.analyze_equ(),
            Directive::Include => self.analyze_include(),
            Directive::Org => self.analyze_org(),
        }
    }

    fn analyze_data(self, width: Width) {
        let session = &mut self.actions.session;
        for arg in self.args {
            let expr = {
                let builder = &mut session.backend.build_value(session.names);
                let mut context = ValueContext::new(builder, session.diagnostics);
                if let Ok(expr) = context.analyze_expr(arg) {
                    expr
                } else {
                    return;
                }
            };
            session.backend.emit_item(backend::Item::Data(expr, width))
        }
    }

    fn analyze_ds(self) {
        let session = &mut self.actions.session;
        let origin = {
            let arg = if let Ok(arg) = single_arg(self.span, self.args, session.diagnostics) {
                arg
            } else {
                return;
            };
            let builder = &mut session.backend.build_value(session.names);
            let mut context = ValueContext::new(builder, session.diagnostics);
            let count = if let Ok(count) = context.analyze_expr(arg) {
                count
            } else {
                return;
            };
            location_counter_plus_expr(count, builder)
        };
        session.backend.set_origin(origin)
    }

    fn analyze_equ(self) {
        let session = &mut self.actions.session;
        let symbol = self.actions.label.take().unwrap();
        let arg = if let Ok(arg) = single_arg(self.span, self.args, session.diagnostics) {
            arg
        } else {
            return;
        };

        let value = {
            let builder = &mut session.backend.build_value(session.names);
            let mut context = ValueContext::new(builder, session.diagnostics);
            if let Ok(value) = context.analyze_expr(arg) {
                value
            } else {
                return;
            }
        };
        session.backend.define_symbol(symbol, value, session.names)
    }

    fn analyze_include(self) {
        let (path, span) = if let Ok(result) =
            reduce_include(self.span, self.args, self.actions.session.diagnostics)
        {
            result
        } else {
            return;
        };
        if let Err(err) = self.actions.session.analyze_file(path) {
            self.actions
                .diagnostics()
                .emit_diagnostic(CompactDiagnostic::new(err.into(), span))
        }
    }

    fn analyze_org(self) {
        let session = &mut self.actions.session;
        let arg = if let Ok(arg) = single_arg(self.span, self.args, session.diagnostics) {
            arg
        } else {
            return;
        };
        let expr = {
            let builder = &mut session.backend.build_value(session.names);
            let mut context = ValueContext::new(builder, session.diagnostics);
            if let Ok(expr) = context.analyze_expr(arg) {
                expr
            } else {
                return;
            }
        };
        session.backend.set_origin(expr)
    }
}

fn location_counter_plus_expr<I, B, S>(expr: B::Value, builder: &mut B) -> B::Value
where
    B: ValueBuilder<I, S>,
    S: Clone,
{
    let location = builder.to_value((LocationCounter, expr.span()));
    builder.apply_binary_operator((BinaryOperator::Plus, expr.span()), location, expr)
}

fn reduce_include<I: PartialEq, D: DownstreamDiagnostics<S>, S>(
    span: S,
    args: Vec<SemanticExpr<I, S>>,
    diagnostics: &mut D,
) -> Result<(I, S), ()> {
    let arg = single_arg(span, args, diagnostics)?;
    match arg.variant {
        ExprVariant::Atom(SemanticAtom::Literal(Literal::String(path))) => Ok((path, arg.span)),
        _ => {
            diagnostics.emit_diagnostic(CompactDiagnostic::new(Message::ExpectedString, arg.span));
            Err(())
        }
    }
}

fn single_arg<T, D: DownstreamDiagnostics<S>, S>(
    span: S,
    args: impl IntoIterator<Item = T>,
    diagnostics: &mut D,
) -> Result<T, ()> {
    let mut args = args.into_iter();
    if let Some(arg) = args.next() {
        assert!(args.next().is_none());
        Ok(arg)
    } else {
        diagnostics.emit_diagnostic(CompactDiagnostic::new(
            Message::OperandCount {
                actual: 0,
                expected: 1,
            },
            span,
        ));
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{NameTable, RelocAtom, RelocExpr};
    use crate::codebase::CodebaseError;
    use crate::diag;
    use crate::frontend::semantics;
    use crate::frontend::semantics::tests::{MockDiagnostics, *};
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

    fn mk_byte(byte: &i32) -> backend::Item<RelocExpr<Ident<String>, ()>> {
        backend::Item::Data((*byte).into(), Width::Byte)
    }

    fn mk_word(word: &i32) -> backend::Item<RelocExpr<Ident<String>, ()>> {
        backend::Item::Data((*word).into(), Width::Word)
    }

    fn test_data_items_emission(
        directive: Directive,
        mk_item: impl Fn(&i32) -> backend::Item<RelocExpr<Ident<String>, ()>>,
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

    fn mk_literal(n: i32) -> (ExprAtom<Ident<String>, Literal<String>>, ()) {
        (ExprAtom::Literal(Literal::Number(n)), ())
    }

    #[test]
    fn ds_with_malformed_expr() {
        let actions =
            ds(|arg| arg.push_atom((ExprAtom::Literal(Literal::Operand(Operand::A)), ())));
        assert_eq!(
            actions,
            [diag::Event::EmitDiagnostic(CompactDiagnostic::new(
                Message::KeywordInExpr { keyword: () },
                (),
            ))
            .into()]
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
            [
                diag::Event::EmitDiagnostic(CompactDiagnostic::new(Message::ExpectedString, (),))
                    .into()
            ]
        )
    }

    #[test]
    fn data_with_malformed_expr() {
        let actions = unary_directive(Directive::Db, |arg| {
            arg.push_atom((ExprAtom::Literal(Literal::Operand(Operand::A)), ()))
        });
        assert_eq!(
            actions,
            [diag::Event::EmitDiagnostic(CompactDiagnostic::new(
                Message::KeywordInExpr { keyword: () },
                (),
            ))
            .into()]
        )
    }

    #[test]
    fn include_file_with_invalid_utf8() {
        let name = "invalid_utf8.s";
        let operations = RefCell::new(Vec::new());
        let mut frontend = TestFrontend::new(&operations);
        frontend.fail(CodebaseError::Utf8Error);
        let mut backend = TestBackend::new(&operations);
        let mut names = NameTable::new();
        let mut diagnostics = MockDiagnostics::new(&operations);
        let session = Session::new(&mut frontend, &mut backend, &mut names, &mut diagnostics);
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
                diag::Event::EmitDiagnostic(CompactDiagnostic::new(Message::InvalidUtf8, ()))
                    .into()
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
        let mut names = NameTable::new();
        let mut diagnostics = MockDiagnostics::new(&operations);
        let session = Session::new(&mut frontend, &mut backend, &mut names, &mut diagnostics);
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
                diag::Event::EmitDiagnostic(CompactDiagnostic::new(
                    Message::IoError {
                        string: message.to_string()
                    },
                    ()
                ))
                .into()
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
            &mut semantics::ExprContext<'a, TestFrontend<'a>, TestBackend<'a>, MockDiagnostics<'a>>,
        ),
    ) -> Vec<TestOperation> {
        unary_directive(Directive::Ds, f)
    }

    type TestExprContext<'a> =
        semantics::ExprContext<'a, TestFrontend<'a>, TestBackend<'a>, MockDiagnostics<'a>>;

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
            [diag::Event::EmitDiagnostic(CompactDiagnostic::new(
                Message::OperandCount {
                    actual: 0,
                    expected: 1
                },
                (),
            ))
            .into()]
        )
    }

    type TestCommandActions<'a> =
        semantics::CommandActions<'a, TestFrontend<'a>, TestBackend<'a>, MockDiagnostics<'a>>;

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
