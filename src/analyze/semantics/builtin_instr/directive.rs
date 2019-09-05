use super::{DefaultBuiltinInstrSet, UnboundBuiltinInstrMnemonic};

use crate::analyze::reentrancy::ReentrancyActions;
use crate::analyze::semantics::arg::*;
use crate::analyze::semantics::params::RelocLookup;
use crate::analyze::semantics::resolve::{NameTable, StartScope};
use crate::analyze::semantics::*;
use crate::analyze::semantics::{Keyword, Session};
use crate::analyze::Literal;
use crate::diag::*;
use crate::object::builder::{Backend, Item, Width};

use std::ops::DerefMut;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum Directive {
    Binding(BindingDirective),
    Simple(SimpleDirective),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::analyze) enum BindingDirective {
    Equ,
    Macro,
    Section,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::analyze) enum SimpleDirective {
    Db,
    Ds,
    Dw,
    Endc,
    If,
    Include,
    Org,
}

pub(in crate::analyze::semantics) fn analyze_directive<R, N, B>(
    directive: (Directive, R::Span),
    label: Option<Label<R::Ident, R::Span>>,
    args: BuiltinInstrArgs<R::Ident, R::StringRef, R::Span>,
    session: TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B>,
) -> TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword<BindingDirective, UnboundBuiltinInstrMnemonic>,
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

struct DirectiveContext<R: ReentrancyActions, N, B> {
    span: R::Span,
    label: Option<Label<R::Ident, R::Span>>,
    args: BuiltinInstrArgs<R::Ident, R::StringRef, R::Span>,
    session: TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B>,
}

impl<R, N, B> DirectiveContext<R, N, B>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword<BindingDirective, UnboundBuiltinInstrMnemonic>,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: Backend<R::Span>,
{
    fn analyze(
        self,
        directive: Directive,
    ) -> TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B> {
        use self::BindingDirective::*;
        use self::SimpleDirective::*;
        match directive {
            Directive::Binding(Equ) => self.analyze_equ(),
            Directive::Binding(Macro) => self.analyze_macro(),
            Directive::Binding(Section) => self.analyze_section(),
            Directive::Simple(Db) => self.analyze_data(Width::Byte),
            Directive::Simple(Ds) => self.analyze_ds(),
            Directive::Simple(Dw) => self.analyze_data(Width::Word),
            Directive::Simple(Endc) => self.analyze_endc(),
            Directive::Simple(If) => self.analyze_if(),
            Directive::Simple(Include) => self.analyze_include(),
            Directive::Simple(Org) => self.analyze_org(),
        }
    }

    fn analyze_data(
        mut self,
        width: Width,
    ) -> TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B> {
        for arg in self.args {
            let expr = match self.session.analyze_expr(arg) {
                (Ok(expr), session) => {
                    self.session = session;
                    expr
                }
                (Err(()), session) => return session,
            };
            self.session.builder.emit_item(Item::Data(expr, width))
        }
        self.session
    }

    fn analyze_ds(mut self) -> TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B> {
        match single_arg(self.span, self.args, &mut self.session) {
            Ok(arg) => {
                let (result, session) = self.session.analyze_expr(arg);
                self.session = session;
                if let Ok(bytes) = result {
                    self.session.builder.reserve(bytes)
                }
            }
            Err(()) => (),
        }
        self.session
    }

    fn analyze_equ(mut self) -> TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B> {
        let (symbol, params) = self.label.take().unwrap();
        match single_arg(self.span, self.args, &mut self.session) {
            Ok(arg) => {
                let (_, session) = self.session.define_symbol_with_params(symbol, &params, arg);
                self.session = session;
            }
            Err(()) => (),
        }
        self.session
    }

    fn analyze_section(mut self) -> TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B> {
        let (name, span) = self.label.take().unwrap().0;
        let id = self.session.reloc_lookup(name, span.clone());
        self.session.builder.start_section(id, span);
        self.session
    }

    fn analyze_endc(self) -> TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B> {
        self.session
    }

    fn analyze_if(mut self) -> TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B> {
        match single_arg(self.span, self.args, &mut self.session) {
            Ok(arg) => {
                let (value, session) = self.session.analyze_expr(arg);
                self.session = session;
                match self
                    .session
                    .builder
                    .is_non_zero(value.unwrap(), &mut self.session.reentrancy)
                {
                    Some(true) => (),
                    Some(false) => {
                        self.session.state.current = LineRule::TokenLine(TokenContext::FalseIf)
                    }
                    None => unimplemented!(),
                }
            }
            Err(()) => unimplemented!(),
        }
        self.session
    }

    fn analyze_include(mut self) -> TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B> {
        let (path, span) = match reduce_include(self.span, self.args, &mut self.session) {
            Ok(result) => result,
            Err(()) => return self.session,
        };
        let (result, mut semantics): (_, TokenStreamSemantics<_, _, _, _>) =
            self.session.reentrancy.analyze_file(
                path,
                Session {
                    instr_set: self.session.instr_set,
                    reentrancy: (),
                    names: self.session.names,
                    builder: self.session.builder,
                    state: self.session.state,
                    stack: self.session.stack,
                },
            );
        if let Err(err) = result {
            semantics.emit_diag(Message::from(err).at(span))
        }
        semantics
    }

    fn analyze_macro(mut self) -> TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B> {
        if self.label.is_none() {
            let span = self.span;
            self.session.emit_diag(Message::MacroRequiresName.at(span))
        }
        set_state!(
            self.session,
            TokenContext::MacroDef(MacroDefState::new(self.label)).into()
        )
    }

    fn analyze_org(mut self) -> TokenStreamSemantics<DefaultBuiltinInstrSet, R, N, B> {
        match single_arg(self.span, self.args, &mut self.session) {
            Ok(arg) => {
                let (result, session) = self.session.analyze_expr(arg);
                self.session = session;
                if let Ok(value) = result {
                    self.session.builder.set_origin(value)
                }
            }
            Err(()) => (),
        }
        self.session
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

    use crate::analyze::macros::mock::MockMacroId;
    use crate::analyze::reentrancy::ReentrancyEvent;
    use crate::analyze::semantics::actions::tests::*;
    use crate::analyze::semantics::resolve::{MockNameTable, NameTableEvent, ResolvedName};
    use crate::analyze::syntax::actions::*;
    use crate::analyze::syntax::Sigil;
    use crate::codebase::CodebaseError;
    use crate::expr::{Atom, ParamId};
    use crate::object::builder::mock::*;
    use crate::object::builder::{CpuInstr, Nullary};

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
    fn dispatch_instrs_in_taken_if() {
        let actions = collect_semantic_actions(|actions| {
            let mut condition = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("IF".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            condition.act_on_atom(ExprAtom::Literal(Literal::Number(1)), ());
            condition
                .did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("NOP".into(), ())
                .into_builtin_instr()
                .did_parse_instr()
                .did_parse_line(())
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("ENDC".into(), ())
                .into_builtin_instr()
                .did_parse_instr()
                .did_parse_line(())
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("STOP".into(), ())
                .into_builtin_instr()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [
                BackendEvent::EmitItem(Item::CpuInstr(CpuInstr::Nullary(Nullary::Nop))).into(),
                BackendEvent::EmitItem(Item::CpuInstr(CpuInstr::Nullary(Nullary::Stop))).into()
            ]
        )
    }

    #[test]
    fn ignore_instrs_in_untaken_if() {
        let actions = collect_semantic_actions(|actions| {
            let mut condition = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("IF".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            condition.act_on_atom(ExprAtom::Literal(Literal::Number(0)), ());
            let mut then = condition
                .did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .will_parse_line()
                .into_token_line()
                .act_on_ident("NOP".into(), ())
                .into_token_seq();
            then.act_on_token(Sigil::Eol.into(), ());
            then.act_on_ident("ENDC".into(), ())
                .into_line_end()
                .did_parse_line(())
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("STOP".into(), ())
                .into_builtin_instr()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [BackendEvent::EmitItem(Item::CpuInstr(CpuInstr::Nullary(Nullary::Stop))).into()]
        )
    }

    fn ds(f: impl FnOnce(&mut TestExprContext<()>)) -> Vec<TestOperation<()>> {
        unary_directive("DS", f)
    }

    type TestExprContext<S> = ArgSemantics<
        DefaultBuiltinInstrSet,
        MockSourceComponents<S>,
        Box<
            MockNameTable<
                BasicNameTable<
                    &'static Keyword<BindingDirective, UnboundBuiltinInstrMnemonic>,
                    MockMacroId,
                    MockSymbolId,
                >,
                TestOperation<S>,
            >,
        >,
        MockBackend<SerialIdAllocator<MockSymbolId>, TestOperation<S>>,
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

    type TestBuiltinInstrSemantics<S> = BuiltinInstrSemantics<
        DefaultBuiltinInstrSet,
        MockSourceComponents<S>,
        Box<
            MockNameTable<
                BasicNameTable<
                    &'static Keyword<BindingDirective, UnboundBuiltinInstrMnemonic>,
                    MockMacroId,
                    MockSymbolId,
                >,
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
