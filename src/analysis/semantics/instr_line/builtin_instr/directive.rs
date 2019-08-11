use super::*;

use crate::analysis::semantics::instr_line::Label;
use crate::analysis::semantics::params::RelocLookup;
use crate::analysis::semantics::token_line::{MacroDefState, TokenContext, TokenLineSemantics};
use crate::analysis::semantics::TokenStreamState;
use crate::analysis::session::Session;
use crate::analysis::Literal;
use crate::diag::*;
use crate::model::{Item, Width};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::analysis) enum Directive {
    Db,
    Ds,
    Dw,
    Equ,
    Include,
    Macro,
    Org,
    Section,
}

pub(super) fn analyze_directive<S: Session>(
    directive: (Directive, S::Span),
    label: Option<Label<S::Ident, S::Span>>,
    args: BuiltinInstrArgs<S::Ident, S::StringRef, S::Span>,
    actions: InstrLineSemantics<S>,
) -> TokenStreamSemantics<S> {
    let context = DirectiveContext {
        span: directive.1,
        label,
        args,
        actions,
    };
    context.analyze(directive.0)
}

struct DirectiveContext<A, I, R, S> {
    span: S,
    label: Option<Label<I, S>>,
    args: BuiltinInstrArgs<I, R, S>,
    actions: A,
}

delegate_diagnostics! {
    {S: Session},
    DirectiveContext<InstrLineSemantics<S>, S::Ident, S::StringRef, S::Span>,
    {actions},
    InstrLineSemantics<S>,
    S::Span
}

impl<'a, S: Session> DirectiveContext<InstrLineSemantics<S>, S::Ident, S::StringRef, S::Span> {
    fn analyze(self, directive: Directive) -> TokenStreamSemantics<S> {
        match directive {
            Directive::Db => self.analyze_data(Width::Byte),
            Directive::Ds => self.analyze_ds(),
            Directive::Dw => self.analyze_data(Width::Word),
            Directive::Equ => self.analyze_equ(),
            Directive::Include => self.analyze_include(),
            Directive::Macro => self.analyze_macro(),
            Directive::Org => self.analyze_org(),
            Directive::Section => self.analyze_section(),
        }
    }

    fn analyze_data(mut self, width: Width) -> TokenStreamSemantics<S> {
        for arg in self.args {
            let expr = match self.actions.analyze_expr(&Default::default(), arg) {
                (Ok(expr), actions) => {
                    self.actions = actions;
                    expr
                }
                (Err(()), actions) => return actions.map_line(Into::into),
            };
            self.actions.session.emit_item(Item::Data(expr, width))
        }
        self.actions.map_line(Into::into)
    }

    fn analyze_ds(self) -> TokenStreamSemantics<S> {
        let mut actions = self.actions;
        match single_arg(self.span, self.args, &mut actions) {
            Ok(arg) => {
                let (result, returned_actions) = actions.analyze_expr(&Default::default(), arg);
                actions = returned_actions;
                if let Ok(bytes) = result {
                    actions.session.reserve(bytes)
                }
            }
            Err(()) => (),
        }
        actions.map_line(Into::into)
    }

    fn analyze_equ(mut self) -> TokenStreamSemantics<S> {
        let (symbol, params) = self.label.take().unwrap();
        match single_arg(self.span, self.args, &mut self.actions) {
            Ok(arg) => {
                let (_, actions) = self.actions.define_symbol(symbol, &params, arg);
                self.actions = actions;
            }
            Err(()) => (),
        }
        self.actions.map_line(Into::into)
    }

    fn analyze_section(mut self) -> TokenStreamSemantics<S> {
        let (name, span) = self.label.take().unwrap().0;
        let session = &mut self.actions.session;
        let id = session.reloc_lookup(name, span.clone());
        session.start_section((id, span));
        self.actions.map_line(Into::into)
    }

    fn analyze_include(mut self) -> TokenStreamSemantics<S> {
        let (path, span) = match reduce_include(self.span, self.args, &mut self.actions) {
            Ok(result) => result,
            Err(()) => return self.actions.map_line(Into::into),
        };
        let (result, mut semantics): (_, TokenStreamSemantics<_>) = self
            .actions
            .session
            .analyze_file(path, TokenStreamState::from(self.actions.state));
        if let Err(err) = result {
            semantics.emit_diag(Message::from(err).at(span))
        }
        semantics
    }

    fn analyze_macro(mut self) -> TokenStreamSemantics<S> {
        if self.label.is_none() {
            let span = self.span;
            self.actions.emit_diag(Message::MacroRequiresName.at(span))
        }
        TokenLineSemantics {
            state: TokenContext::MacroDef(MacroDefState::new(self.label)),
            session: self.actions.session,
        }
        .map_line(Into::into)
    }

    fn analyze_org(self) -> TokenStreamSemantics<S> {
        let mut actions = self.actions;
        match single_arg(self.span, self.args, &mut actions) {
            Ok(arg) => {
                let (result, returned_actions) = actions.analyze_expr(&Default::default(), arg);
                actions = returned_actions;
                if let Ok(value) = result {
                    actions.session.set_origin(value)
                }
            }
            Err(()) => (),
        }
        actions.map_line(Into::into)
    }
}

fn reduce_include<I: PartialEq, R, D: Diagnostics<S>, S>(
    span: S,
    args: Vec<Arg<I, R, S>>,
    diagnostics: &mut D,
) -> Result<(R, S), ()> {
    let arg = single_arg(span, args, diagnostics)?;
    match arg.variant {
        ArgVariant::Atom(ArgAtom::Literal(Literal::String(path))) => Ok((path, arg.span)),
        _ => {
            diagnostics.emit_diag(Message::ExpectedString.at(arg.span));
            Err(())
        }
    }
}

fn single_arg<T, D: Diagnostics<S>, S>(
    span: S,
    args: impl IntoIterator<Item = T>,
    diagnostics: &mut D,
) -> Result<T, ()> {
    let mut args = args.into_iter();
    if let Some(arg) = args.next() {
        assert!(args.next().is_none());
        Ok(arg)
    } else {
        diagnostics.emit_diag(
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
    use crate::analysis::resolve::{NameTableEvent, ResolvedIdent};
    use crate::analysis::semantics::instr_line::builtin_instr;
    use crate::analysis::semantics::tests::*;
    use crate::analysis::session::SessionEvent;
    use crate::analysis::syntax::actions::*;
    use crate::codebase::CodebaseError;
    use crate::log::with_log;
    use crate::model::{Atom, ParamId};

    use std::borrow::Borrow;
    use std::io;

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let actions = unary_directive("INCLUDE", |arg| {
            arg.act_on_atom(ExprAtom::Literal(Literal::String(filename.to_string())), ());
        });
        assert_eq!(
            actions,
            [SessionEvent::AnalyzeFile(filename.to_string()).into()]
        )
    }

    #[test]
    fn set_origin() {
        let origin = 0x3000;
        let actions = unary_directive("ORG", |arg| arg.act_on_atom(mk_literal(origin), ()));
        assert_eq!(actions, [BackendEvent::SetOrigin(origin.into()).into()])
    }

    #[test]
    fn emit_byte_items() {
        test_data_items_emission("DB", mk_byte, [0x42, 0x78])
    }

    #[test]
    fn emit_word_items() {
        test_data_items_emission("DW", mk_word, [0x4332, 0x780f])
    }

    fn mk_byte(byte: i32) -> Item<Expr<()>> {
        Item::Data((byte).into(), Width::Byte)
    }

    fn mk_word(word: i32) -> Item<Expr<()>> {
        Item::Data((word).into(), Width::Word)
    }

    fn test_data_items_emission(
        directive: &str,
        mk_item: impl Fn(i32) -> Item<Expr<()>>,
        data: impl Borrow<[i32]>,
    ) {
        let actions = with_directive(directive, |mut command| {
            for datum in data.borrow().iter() {
                let mut arg = command.will_parse_arg();
                arg.act_on_atom(mk_literal(*datum), ());
                command = arg.did_parse_arg();
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
        let actions = ds(|arg| arg.act_on_atom(mk_literal(3), ()));
        assert_eq!(actions, [BackendEvent::Reserve(3.into()).into()])
    }

    fn mk_literal(n: i32) -> ExprAtom<String, Literal<String>> {
        ExprAtom::Literal(Literal::Number(n))
    }

    #[test]
    fn ds_with_malformed_expr() {
        let actions = ds(|arg| arg.act_on_atom(ExprAtom::Ident("A".into()), ()));
        assert_eq!(
            actions,
            [
                DiagnosticsEvent::EmitDiag(Message::KeywordInExpr { keyword: () }.at(()).into())
                    .into()
            ]
        )
    }

    #[test]
    fn ds_without_args() {
        test_unary_directive_without_args("DS")
    }

    #[test]
    fn org_without_args() {
        test_unary_directive_without_args("ORG")
    }

    #[test]
    fn include_without_args() {
        test_unary_directive_without_args("INCLUDE")
    }

    #[test]
    fn include_with_number() {
        let actions = unary_directive("INCLUDE", |arg| arg.act_on_atom(mk_literal(7), ()));
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiag(Message::ExpectedString.at(()).into()).into()]
        )
    }

    #[test]
    fn data_with_malformed_expr() {
        let actions = unary_directive("DB", |arg| arg.act_on_atom(ExprAtom::Ident("A".into()), ()));
        assert_eq!(
            actions,
            [
                DiagnosticsEvent::EmitDiag(Message::KeywordInExpr { keyword: () }.at(()).into())
                    .into()
            ]
        )
    }

    #[test]
    fn include_file_with_invalid_utf8() {
        let name = "invalid_utf8.s";
        let log = with_log(|log| {
            let mut session = MockSession::with_log(log);
            session.fail(CodebaseError::Utf8Error);
            let mut context = TokenStreamSemantics::new(session)
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("INCLUDE".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            context.act_on_atom(ExprAtom::Literal(Literal::String(name.into())), ());
            context.did_parse_arg().did_parse_instr().did_parse_line(());
        });
        assert_eq!(
            log,
            [
                SessionEvent::AnalyzeFile(name.into()).into(),
                DiagnosticsEvent::EmitDiag(Message::InvalidUtf8.at(()).into()).into()
            ]
        )
    }

    #[test]
    fn include_nonexistent_file() {
        let name = "nonexistent.s";
        let message = "some message";
        let log = with_log(|log| {
            let mut session = MockSession::with_log(log);
            session.fail(CodebaseError::IoError(io::Error::new(
                io::ErrorKind::NotFound,
                message,
            )));
            let mut context = TokenStreamSemantics::new(session)
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("INCLUDE".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            context.act_on_atom(ExprAtom::Literal(Literal::String(name.into())), ());
            context.did_parse_arg().did_parse_instr().did_parse_line(());
        });
        assert_eq!(
            log,
            [
                SessionEvent::AnalyzeFile(name.into()).into(),
                DiagnosticsEvent::EmitDiag(
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
        let actions = with_labeled_directive(symbol, "EQU", |arg| {
            arg.act_on_atom(ExprAtom::Literal(Literal::Number(value)), ())
        });
        assert_eq!(
            actions,
            [
                NameTableEvent::Insert(symbol.into(), ResolvedIdent::Backend(0)).into(),
                BackendEvent::DefineSymbol((0, ()), value.into()).into()
            ]
        )
    }

    #[test]
    fn define_fn_with_param() {
        let name = "my_fn";
        let param = "param";
        let actions = collect_semantic_actions(|builder| {
            let mut label_actions = builder
                .will_parse_line()
                .into_instr_line()
                .will_parse_label((name.into(), ()));
            label_actions.act_on_param(param.into(), ());
            let mut arg_actions = label_actions
                .did_parse_label()
                .will_parse_instr("EQU".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            arg_actions.act_on_atom(ExprAtom::Ident(param.into()), ());
            arg_actions
                .did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [
                NameTableEvent::Insert(name.into(), ResolvedIdent::Backend(0)).into(),
                BackendEvent::DefineSymbol((0, ()), Atom::from(ParamId(0)).into()).into()
            ]
        )
    }

    #[test]
    fn start_section() {
        let name = "hot_stuff";
        let actions = collect_semantic_actions(|actions| {
            actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_label((name.into(), ()))
                .did_parse_label()
                .will_parse_instr("SECTION".into(), ())
                .into_builtin_instr()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [
                NameTableEvent::Insert(name.into(), ResolvedIdent::Backend(0)).into(),
                BackendEvent::StartSection((0, ())).into()
            ]
        )
    }

    fn ds(f: impl FnOnce(&mut TestExprContext<()>)) -> Vec<TestOperation<()>> {
        unary_directive("DS", f)
    }

    type TestExprContext<S> = builtin_instr::ArgSemantics<MockSession<S>>;

    fn unary_directive<F>(directive: &str, f: F) -> Vec<TestOperation<()>>
    where
        F: FnOnce(&mut TestExprContext<()>),
    {
        with_directive(directive, |command| {
            let mut arg = command.will_parse_arg();
            f(&mut arg);
            arg.did_parse_arg()
        })
    }

    fn test_unary_directive_without_args(directive: &str) {
        let actions = with_directive(directive, |command| command);
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiag(
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

    type TestBuiltinInstrSemantics<S> = builtin_instr::BuiltinInstrSemantics<MockSession<S>>;

    fn with_directive<F>(directive: &str, f: F) -> Vec<TestOperation<()>>
    where
        F: FnOnce(TestBuiltinInstrSemantics<()>) -> TestBuiltinInstrSemantics<()>,
    {
        collect_semantic_actions(|actions| {
            let command = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr(directive.into(), ())
                .into_builtin_instr();
            f(command)
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        })
    }

    fn with_labeled_directive<F>(label: &str, directive: &str, f: F) -> Vec<TestOperation<()>>
    where
        F: FnOnce(&mut TestExprContext<()>),
    {
        collect_semantic_actions(|actions| {
            let mut arg = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_label((label.into(), ()))
                .did_parse_label()
                .will_parse_instr(directive.into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            f(&mut arg);
            arg.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        })
    }
}
