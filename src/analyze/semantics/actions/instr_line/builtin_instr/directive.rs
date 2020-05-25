use crate::analyze::semantics::arg::*;
use crate::analyze::semantics::keywords::Directive;
use crate::analyze::semantics::params::RelocLookup;
use crate::analyze::semantics::reentrancy::ReentrancyActions;
use crate::analyze::semantics::resolve::{NameTable, StartScope};
use crate::analyze::semantics::*;
use crate::analyze::semantics::{Keyword, Semantics};
use crate::diag::span::Source;
use crate::diag::*;
use crate::object::builder::{Backend, Item, Width};

use std::ops::DerefMut;

pub(in crate::analyze::semantics) fn analyze_directive<R, N, B>(
    directive: (Directive, R::Span),
    label: Option<Label<R::Ident, R::Span>>,
    args: BuiltinInstrArgs<B::Value, R::StringRef, R::Span>,
    session: TokenStreamSemantics<R, N, B>,
) -> TokenStreamSemantics<R, N, B>
where
    R: ReentrancyActions,
    R::Ident: 'static,
    R::StringRef: 'static,
    R::Span: 'static,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: Backend<R::Span>,
{
    let context = DirectiveContext {
        span: directive.1,
        label,
        args,
        session,
    };
    context.analyze(directive.0)
}

struct DirectiveContext<'a, R: ReentrancyActions, N, B: PartialBackend<R::Span>> {
    span: R::Span,
    label: Option<Label<R::Ident, R::Span>>,
    args: BuiltinInstrArgs<B::Value, R::StringRef, R::Span>,
    session: TokenStreamSemantics<'a, R, N, B>,
}

impl<'a, R, N, B> DirectiveContext<'a, R, N, B>
where
    R: ReentrancyActions,
    R::Ident: 'static,
    R::StringRef: 'static,
    R::Span: 'static,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: Backend<R::Span>,
{
    fn analyze(self, directive: Directive) -> TokenStreamSemantics<'a, R, N, B> {
        match directive {
            Directive::Equ => self.analyze_equ(),
            Directive::Macro => self.analyze_macro(),
            Directive::Section => self.analyze_section(),
            Directive::Db => self.analyze_data(Width::Byte),
            Directive::Ds => self.analyze_ds(),
            Directive::Dw => self.analyze_data(Width::Word),
            Directive::Endc => self.analyze_endc(),
            Directive::Endm => unimplemented!(),
            Directive::If => self.analyze_if(),
            Directive::Include => self.analyze_include(),
            Directive::Org => self.analyze_org(),
        }
    }

    fn analyze_data(mut self, width: Width) -> TokenStreamSemantics<'a, R, N, B> {
        for arg in self.args {
            let expr = match self.session.expect_const(arg) {
                Ok(expr) => expr,
                Err(()) => return self.session,
            };
            self.session.core.builder.emit_item(Item::Data(expr, width))
        }
        self.session
    }

    fn analyze_ds(mut self) -> TokenStreamSemantics<'a, R, N, B> {
        if let Some(arg) = single_arg(self.span, self.args, &mut self.session) {
            let result = self.session.expect_const(arg);
            if let Ok(bytes) = result {
                self.session.core.builder.reserve(bytes)
            }
        }
        self.session
    }

    fn analyze_equ(mut self) -> TokenStreamSemantics<'a, R, N, B> {
        let (symbol, _) = self.label.take().unwrap();
        if let Some(arg) = single_arg(self.span, self.args, &mut self.session) {
            self.session.define_symbol_with_params(symbol, arg);
        }
        self.session
    }

    fn analyze_section(mut self) -> TokenStreamSemantics<'a, R, N, B> {
        let (name, span) = self.label.take().unwrap().0;
        let id = self.session.reloc_lookup(name, span.clone());
        self.session.core.builder.start_section(id, span);
        self.session
    }

    fn analyze_endc(self) -> TokenStreamSemantics<'a, R, N, B> {
        self.session
    }

    fn analyze_if(mut self) -> TokenStreamSemantics<'a, R, N, B> {
        match single_arg(self.span, self.args, &mut self.session) {
            Some(arg) => {
                let value = self.session.expect_const(arg);
                match self
                    .session
                    .core
                    .builder
                    .is_non_zero(value.unwrap(), &mut self.session.reentrancy)
                {
                    Some(true) => (),
                    Some(false) => {
                        self.session.state.mode = LineRule::TokenLine(TokenLineState {
                            context: TokenContext::FalseIf,
                        })
                    }
                    None => unimplemented!(),
                }
            }
            None => unimplemented!(),
        }
        self.session
    }

    fn analyze_include(mut self) -> TokenStreamSemantics<'a, R, N, B> {
        let (path, span) = match reduce_include(self.span, self.args, &mut self.session) {
            Some(result) => result,
            None => return self.session,
        };
        let (result, (mut reentrancy, core)) = self
            .session
            .reentrancy
            .analyze_file(path, self.session.core);
        if let Err(err) = result {
            reentrancy.emit_diag(Message::from(err).at(span))
        }
        Semantics {
            reentrancy,
            core,
            state: TokenStreamState::new(),
            tokens: self.session.tokens,
        }
    }

    fn analyze_macro(mut self) -> TokenStreamSemantics<'a, R, N, B> {
        if self.label.is_none() {
            let span = self.span;
            self.session.emit_diag(Message::MacroRequiresName.at(span))
        }
        set_state!(
            self.session,
            TokenLineState {
                context: TokenContext::MacroDef(MacroDefState::new(self.label))
            }
            .into()
        )
    }

    fn analyze_org(mut self) -> TokenStreamSemantics<'a, R, N, B> {
        if let Some(arg) = single_arg(self.span, self.args, &mut self.session) {
            let result = self.session.expect_const(arg);
            if let Ok(value) = result {
                self.session.core.builder.set_origin(value)
            }
        }
        self.session
    }
}

fn reduce_include<V: Source<Span = S>, R, D: Diagnostics<S>, S: Clone>(
    span: S,
    args: Vec<Arg<V, R, S>>,
    diagnostics: &mut D,
) -> Option<(R, S)> {
    let arg = match single_arg(span, args, diagnostics) {
        Some(arg) => arg,
        None => return None,
    };
    let result = match arg {
        Arg::String(path, span) => Ok((path, span)),
        Arg::Bare(DerefableArg::Const(value)) => Err(Some(value.span())),
        Arg::Bare(DerefableArg::Symbol(_, span)) => Err(Some(span)),
        Arg::Deref(_, span) => Err(Some(span)),
        Arg::Error => Err(None),
    };
    match result {
        Ok(result) => Some(result),
        Err(Some(span)) => {
            diagnostics.emit_diag(Message::ExpectedString.at(span));
            None
        }
        Err(None) => None,
    }
}

fn single_arg<T, D: Diagnostics<S>, S>(
    span: S,
    args: impl IntoIterator<Item = T>,
    diagnostics: &mut D,
) -> Option<T> {
    let mut args = args.into_iter();
    if let Some(arg) = args.next() {
        assert!(args.next().is_none());
        Some(arg)
    } else {
        diagnostics.emit_diag(
            Message::OperandCount {
                actual: 0,
                expected: 1,
            }
            .at(span),
        );
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analyze::macros::mock::MockMacroId;
    use crate::analyze::semantics::actions::tests::*;
    use crate::analyze::semantics::reentrancy::ReentrancyEvent;
    use crate::analyze::semantics::resolve::{MockNameTable, NameTableEvent, ResolvedName};
    use crate::analyze::syntax::actions::*;
    use crate::analyze::Literal;
    use crate::codebase::CodebaseError;
    use crate::expr::{Atom, ParamId};
    use crate::object::builder::mock::*;

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
            [ReentrancyEvent::AnalyzeFile(filename.to_string()).into()]
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
        let log = collect_semantic_actions(|mut actions| {
            actions.reentrancy.fail(CodebaseError::Utf8Error);
            let mut context = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("INCLUDE".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            context.act_on_atom(ExprAtom::Literal(Literal::String(name.into())), ());
            context.did_parse_arg().did_parse_instr().did_parse_line(())
        });
        assert_eq!(
            log,
            [
                ReentrancyEvent::AnalyzeFile(name.into()).into(),
                DiagnosticsEvent::EmitDiag(Message::InvalidUtf8.at(()).into()).into()
            ]
        )
    }

    #[test]
    fn include_nonexistent_file() {
        let name = "nonexistent.s";
        let message = "some message";
        let log = collect_semantic_actions(|mut actions| {
            actions
                .reentrancy
                .fail(CodebaseError::IoError(io::Error::new(
                    io::ErrorKind::NotFound,
                    message,
                )));
            let mut context = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("INCLUDE".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            context.act_on_atom(ExprAtom::Literal(Literal::String(name.into())), ());
            context.did_parse_arg().did_parse_instr().did_parse_line(())
        });
        assert_eq!(
            log,
            [
                ReentrancyEvent::AnalyzeFile(name.into()).into(),
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
                NameTableEvent::Insert(symbol.into(), ResolvedName::Symbol(MockSymbolId(0))).into(),
                BackendEvent::DefineSymbol((MockSymbolId(0), ()), value.into()).into()
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
                NameTableEvent::Insert(name.into(), ResolvedName::Symbol(MockSymbolId(0))).into(),
                BackendEvent::DefineSymbol((MockSymbolId(0), ()), Atom::from(ParamId(0)).into())
                    .into()
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
                NameTableEvent::Insert(name.into(), ResolvedName::Symbol(MockSymbolId(0))).into(),
                BackendEvent::StartSection(MockSymbolId(0), ()).into()
            ]
        )
    }

    #[test]
    fn taken_if_remains_in_instr_mode() {
        collect_semantic_actions(|session| {
            let session = analyze_directive(
                (Directive::If, ()),
                None,
                vec![Arg::Bare(DerefableArg::Const(Expr::from_atom(
                    1.into(),
                    (),
                )))],
                session,
            );
            assert_eq!(
                session.state,
                TokenStreamState {
                    mode: LineRule::InstrLine(InstrLineState { label: None })
                }
            );
            session
        });
    }

    #[test]
    fn ignore_instrs_in_untaken_if() {
        collect_semantic_actions(|session| {
            let session = analyze_directive(
                (Directive::If, ()),
                None,
                vec![Arg::Bare(DerefableArg::Const(Expr::from_atom(
                    0.into(),
                    (),
                )))],
                session,
            );
            assert_eq!(
                session.state,
                TokenStreamState {
                    mode: LineRule::TokenLine(TokenLineState {
                        context: TokenContext::FalseIf
                    })
                }
            );
            session
        });
    }

    fn ds(f: impl FnOnce(&mut TestExprContext<()>)) -> Vec<TestOperation<()>> {
        unary_directive("DS", f)
    }

    type TestExprContext<'a, S> = ArgSemantics<
        'a,
        MockSourceComponents<S>,
        Box<
            MockNameTable<
                BasicNameTable<&'static Keyword, MockMacroId, MockSymbolId>,
                TestOperation<S>,
            >,
        >,
        RelocContext<MockBackend<SerialIdAllocator<MockSymbolId>, TestOperation<S>>, Expr<S>>,
    >;

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

    type TestBuiltinInstrSemantics<'a, S> = BuiltinInstrSemantics<
        'a,
        MockSourceComponents<S>,
        Box<
            MockNameTable<
                BasicNameTable<&'static Keyword, MockMacroId, MockSymbolId>,
                TestOperation<S>,
            >,
        >,
        MockBackend<SerialIdAllocator<MockSymbolId>, TestOperation<S>>,
    >;

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
