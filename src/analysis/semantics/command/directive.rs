use super::super::Label;
use super::{Arg, ArgAtom, ArgVariant, CommandArgs, Directive, SemanticActions};

use crate::analysis::session::Session;
use crate::analysis::Literal;
use crate::diag::*;
use crate::model::{Item, Width};

pub(super) fn analyze_directive<'a, S: Session>(
    directive: (Directive, S::Span),
    label: Option<Label<S::StringRef, S::Span>>,
    args: CommandArgs<S::StringRef, S::Span>,
    actions: &'a mut SemanticActions<S>,
) {
    let context = DirectiveContext {
        span: directive.1,
        label,
        args,
        actions,
    };
    context.analyze(directive.0)
}

struct DirectiveContext<'a, A, R, S> {
    span: S,
    label: Option<Label<R, S>>,
    args: CommandArgs<R, S>,
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
            Directive::Section => self.analyze_section(),
        }
    }

    fn analyze_data(self, width: Width) {
        for arg in self.args {
            let expr = match self.actions.analyze_expr(&Default::default(), arg) {
                Ok(expr) => expr,
                Err(()) => return,
            };
            self.actions.session().emit_item(Item::Data(expr, width))
        }
    }

    fn analyze_ds(mut self) {
        let actions = &mut self.actions;
        single_arg(self.span, self.args, actions.diagnostics())
            .and_then(|arg| actions.analyze_expr(&Default::default(), arg))
            .map(|bytes| actions.session().reserve(bytes))
            .ok();
    }

    fn analyze_equ(mut self) {
        let actions = &mut self.actions;
        let (symbol, params) = self.label.take().unwrap();
        single_arg(self.span, self.args, actions.diagnostics())
            .and_then(|arg| actions.define_symbol(symbol, &params, arg))
            .ok();
    }

    fn analyze_section(mut self) {
        let name = self.label.take().unwrap().0;
        self.actions.session().start_section(name)
    }

    fn analyze_include(self) {
        let (path, span) = match reduce_include(self.span, self.args, self.actions.diagnostics()) {
            Ok(result) => result,
            Err(()) => return,
        };
        let (result, session) = self.actions.session.take().unwrap().analyze_file(path);
        self.actions.session = Some(session);
        if let Err(err) = result {
            self.actions
                .diagnostics()
                .emit_diagnostic(Message::from(err).at(span))
        }
    }

    fn analyze_org(mut self) {
        let actions = &mut self.actions;
        single_arg(self.span, self.args, actions.diagnostics())
            .and_then(|arg| actions.analyze_expr(&Default::default(), arg))
            .map(|expr| actions.session().set_origin(expr))
            .ok();
    }
}

fn reduce_include<I: PartialEq, D: DownstreamDiagnostics<S>, S>(
    span: S,
    args: Vec<Arg<I, S>>,
    diagnostics: &mut D,
) -> Result<(I, S), ()> {
    let arg = single_arg(span, args, diagnostics)?;
    match arg.variant {
        ArgVariant::Atom(ArgAtom::Literal(Literal::String(path))) => Ok((path, arg.span)),
        _ => {
            diagnostics.emit_diagnostic(Message::ExpectedString.at(arg.span));
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
        diagnostics.emit_diagnostic(
            Message::OperandCount {
                actual: 0,
                expected: 1,
            }
            .at(span),
        );
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::backend::BackendEvent;
    use crate::analysis::semantics::tests::*;
    use crate::analysis::session::SessionEvent;
    use crate::analysis::Ident;
    use crate::codebase::CodebaseError;
    use crate::model::{Atom, LocationCounter, ParamId};
    use crate::syntax::keyword::{Command, Operand};
    use crate::syntax::*;

    use std::borrow::Borrow;
    use std::cell::RefCell;
    use std::io;

    type Expr = crate::model::Expr<Atom<LocationCounter, Ident<String>>, ()>;

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

    fn mk_byte(byte: i32) -> Item<Expr> {
        Item::Data((byte).into(), Width::Byte)
    }

    fn mk_word(word: i32) -> Item<Expr> {
        Item::Data((word).into(), Width::Word)
    }

    fn test_data_items_emission(
        directive: Directive,
        mk_item: impl Fn(i32) -> Item<Expr>,
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
                .cloned()
                .map(mk_item)
                .map(BackendEvent::EmitItem)
                .map(Into::into)
                .collect::<Vec<_>>()
        )
    }

    #[test]
    fn reserve_3_bytes() {
        let actions = ds(|arg| arg.push_atom(mk_literal(3)));
        assert_eq!(actions, [BackendEvent::Reserve(3.into()).into()])
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
            [DiagnosticsEvent::EmitDiagnostic(
                Message::KeywordInExpr { keyword: () }.at(()).into()
            )
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
            [DiagnosticsEvent::EmitDiagnostic(Message::ExpectedString.at(()).into()).into()]
        )
    }

    #[test]
    fn data_with_malformed_expr() {
        let actions = unary_directive(Directive::Db, |arg| {
            arg.push_atom((ExprAtom::Literal(Literal::Operand(Operand::A)), ()))
        });
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiagnostic(
                Message::KeywordInExpr { keyword: () }.at(()).into()
            )
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
                .enter_unlabeled_stmt()
                .enter_command((Command::Directive(Directive::Include), ()))
                .add_argument();
            context.push_atom((ExprAtom::Literal(Literal::String(name.into())), ()));
            context.exit().exit().exit();
        }
        assert_eq!(
            operations.into_inner(),
            [
                SessionEvent::AnalyzeFile(name.into()).into(),
                DiagnosticsEvent::EmitDiagnostic(Message::InvalidUtf8.at(()).into()).into()
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
                .enter_unlabeled_stmt()
                .enter_command((Command::Directive(Directive::Include), ()))
                .add_argument();
            context.push_atom((ExprAtom::Literal(Literal::String(name.into())), ()));
            context.exit().exit().exit();
        }
        assert_eq!(
            operations.into_inner(),
            [
                SessionEvent::AnalyzeFile(name.into()).into(),
                DiagnosticsEvent::EmitDiagnostic(
                    Message::IoError {
                        string: message.to_string()
                    }
                    .at(())
                    .into()
                )
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
            [SessionEvent::DefineSymbol((symbol.into(), ()), value.into()).into()]
        )
    }

    #[test]
    fn define_fn_with_param() {
        let name = "my_fn";
        let param = "param";
        let actions = collect_semantic_actions(|builder| {
            let mut label_actions = builder.enter_labeled_stmt((name.into(), ()));
            label_actions.add_parameter((param.into(), ()));
            let mut arg_actions = label_actions
                .next()
                .enter_command((Directive::Equ.into(), ()))
                .add_argument();
            arg_actions.push_atom((ExprAtom::Ident(param.into()), ()));
            arg_actions.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [SessionEvent::DefineSymbol((name.into(), ()), Atom::from(ParamId(0)).into()).into()]
        )
    }

    #[test]
    fn start_section() {
        let name = "hot_stuff";
        let actions = collect_semantic_actions(|actions| {
            actions
                .enter_labeled_stmt((name.into(), ()))
                .next()
                .enter_command((Command::Directive(Directive::Section), ()))
                .exit()
                .exit()
        });
        assert_eq!(actions, [BackendEvent::StartSection((0, ())).into()])
    }

    fn ds(f: impl for<'a> FnOnce(&mut TestExprContext<'a>)) -> Vec<TestOperation> {
        unary_directive(Directive::Ds, f)
    }

    type TestExprContext<'a> =
        crate::analysis::semantics::command::ExprBuilder<String, (), TestCommandActions<'a>>;

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
            [DiagnosticsEvent::EmitDiagnostic(
                Message::OperandCount {
                    actual: 0,
                    expected: 1
                }
                .at(())
                .into()
            )
            .into()]
        )
    }

    type TestCommandActions<'a> = crate::analysis::semantics::CommandActions<MockSession<'a>>;

    fn with_directive<F>(directive: Directive, f: F) -> Vec<TestOperation>
    where
        F: for<'a> FnOnce(TestCommandActions<'a>) -> TestCommandActions<'a>,
    {
        collect_semantic_actions(|actions| {
            let command = actions
                .enter_unlabeled_stmt()
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
                .enter_labeled_stmt((label.into(), ()))
                .next()
                .enter_command((Command::Directive(directive), ()))
                .add_argument();
            f(&mut arg);
            arg.exit().exit().exit()
        })
    }
}
