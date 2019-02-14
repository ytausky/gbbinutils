use super::{AnalyzeExpr, CommandArgs, Directive, SemanticActions, SemanticAtom, SemanticExpr};
use crate::analysis::backend;
use crate::analysis::backend::Width;
use crate::analysis::session::{Session, ValueBuilder};
use crate::analysis::Literal;
use crate::diag::*;
use crate::expr::{BinaryOperator, ExprVariant};
use crate::span::Source;

pub(super) fn analyze_directive<'a, S: Session>(
    directive: (Directive, S::Span),
    args: CommandArgs<S::StringRef, S::Span>,
    actions: &'a mut SemanticActions<S>,
) {
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

impl<'a, S: Session> DelegateDiagnostics<S::Span>
    for DirectiveContext<'a, SemanticActions<S>, S::StringRef, S::Span>
{
    type Delegate = S::Delegate;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.actions.diagnostics()
    }
}

impl<'a, S: Session> DirectiveContext<'a, SemanticActions<S>, S::StringRef, S::Span> {
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
                if let Ok(expr) = session.analyze_expr(arg) {
                    expr
                } else {
                    return;
                }
            };
            session.emit_item(backend::Item::Data(expr, width))
        }
    }

    fn analyze_ds(self) {
        let session = &mut self.actions.session;
        let origin = {
            let arg = if let Ok(arg) = single_arg(self.span, self.args, session.diagnostics()) {
                arg
            } else {
                return;
            };
            let count = if let Ok(count) = session.analyze_expr(arg) {
                count
            } else {
                return;
            };
            location_counter_plus_expr(count, session)
        };
        session.set_origin(origin)
    }

    fn analyze_equ(self) {
        let session = &mut self.actions.session;
        let symbol = self.actions.label.take().unwrap();
        let arg = if let Ok(arg) = single_arg(self.span, self.args, session.diagnostics()) {
            arg
        } else {
            return;
        };
        let value = {
            if let Ok(value) = session.analyze_expr(arg) {
                value
            } else {
                return;
            }
        };
        session.define_symbol(symbol, value)
    }

    fn analyze_include(self) {
        let (path, span) =
            if let Ok(result) = reduce_include(self.span, self.args, self.actions.diagnostics()) {
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
        let arg = if let Ok(arg) = single_arg(self.span, self.args, session.diagnostics()) {
            arg
        } else {
            return;
        };
        let expr = {
            if let Ok(expr) = session.analyze_expr(arg) {
                expr
            } else {
                return;
            }
        };
        session.set_origin(expr)
    }
}

fn location_counter_plus_expr<I, B, S>(expr: B::Value, builder: &mut B) -> B::Value
where
    B: ValueBuilder<I, S>,
    S: Clone,
{
    let location = builder.from_location_counter(expr.span());
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
    use crate::analysis::backend::{BackendEvent, RelocAtom, RelocExpr};
    use crate::analysis::semantics;
    use crate::analysis::semantics::tests::*;
    use crate::analysis::session::SessionEvent;
    use crate::analysis::Ident;
    use crate::codebase::CodebaseError;
    use crate::syntax::keyword::{Command, Operand};
    use crate::syntax::{CommandContext, ExprAtom, ExprContext, FileContext, StmtContext};
    use std::borrow::Borrow;
    use std::cell::RefCell;
    use std::io;

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let actions = unary_directive(Directive::Include, |arg| {
            arg.push_atom((ExprAtom::Literal(Literal::String(filename.to_string())), ()));
        });
        assert_eq!(
            actions,
            [SessionEvent::AnalyzeFile(filename.to_string()).into()]
        )
    }

    #[test]
    fn set_origin() {
        let origin = 0x3000;
        let actions = unary_directive(Directive::Org, |arg| arg.push_atom(mk_literal(origin)));
        assert_eq!(actions, [BackendEvent::SetOrigin(origin.into()).into()])
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
                .map(BackendEvent::EmitItem)
                .map(Into::into)
                .collect::<Vec<_>>()
        )
    }

    #[test]
    fn reserve_3_bytes() {
        let actions = ds(|arg| arg.push_atom(mk_literal(3)));
        assert_eq!(
            actions,
            [BackendEvent::SetOrigin(
                ExprVariant::Binary(
                    BinaryOperator::Plus,
                    Box::new(RelocAtom::LocationCounter.into()),
                    Box::new(3.into()),
                )
                .into()
            )
            .into()]
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
            [DiagnosticsEvent::EmitDiagnostic(CompactDiagnostic::new(
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
            [DiagnosticsEvent::EmitDiagnostic(
                CompactDiagnostic::new(Message::ExpectedString, (),)
            )
            .into()]
        )
    }

    #[test]
    fn data_with_malformed_expr() {
        let actions = unary_directive(Directive::Db, |arg| {
            arg.push_atom((ExprAtom::Literal(Literal::Operand(Operand::A)), ()))
        });
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiagnostic(CompactDiagnostic::new(
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
        let mut session = MockSession::new(&operations);
        session.fail(CodebaseError::Utf8Error);
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
                SessionEvent::AnalyzeFile(name.into()).into(),
                DiagnosticsEvent::EmitDiagnostic(CompactDiagnostic::new(Message::InvalidUtf8, ()))
                    .into()
            ]
        )
    }

    #[test]
    fn include_nonexistent_file() {
        let name = "nonexistent.s";
        let message = "some message";
        let operations = RefCell::new(Vec::new());
        let mut session = MockSession::new(&operations);
        session.fail(CodebaseError::IoError(io::Error::new(
            io::ErrorKind::NotFound,
            message,
        )));
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
                SessionEvent::AnalyzeFile(name.into()).into(),
                DiagnosticsEvent::EmitDiagnostic(CompactDiagnostic::new(
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
            [BackendEvent::DefineSymbol((symbol.into(), ()), value.into()).into()]
        )
    }

    fn ds(f: impl for<'a> FnOnce(&mut TestExprContext<'a>)) -> Vec<TestOperation> {
        unary_directive(Directive::Ds, f)
    }

    type TestExprContext<'a> = semantics::ExprContext<String, (), TestCommandActions<'a>>;

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
            [DiagnosticsEvent::EmitDiagnostic(CompactDiagnostic::new(
                Message::OperandCount {
                    actual: 0,
                    expected: 1
                },
                (),
            ))
            .into()]
        )
    }

    type TestCommandActions<'a> = semantics::CommandActions<MockSession<'a>>;

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
