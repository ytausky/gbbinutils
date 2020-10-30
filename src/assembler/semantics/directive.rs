use crate::assembler::keywords::Directive;
use crate::assembler::semantics::*;
use crate::diagnostics::*;
use crate::object::{Fragment, Width};
use crate::span::Source;

pub(super) fn analyze_directive<'a, S: Analysis>(
    directive: (Directive, S::Span),
    label: Option<Label<S::Span>>,
    args: BuiltinInstrArgs<S::Span>,
    session: TokenStreamSemantics<'a, S>,
) -> TokenStreamSemantics<'a, S>
where
    S::Span: 'static,
{
    let context = DirectiveContext {
        span: directive.1,
        label,
        args,
        session,
    };
    context.analyze(directive.0)
}

struct DirectiveContext<'a, S: Analysis> {
    span: S::Span,
    label: Option<Label<S::Span>>,
    args: BuiltinInstrArgs<S::Span>,
    session: TokenStreamSemantics<'a, S>,
}

impl<'a, S: Analysis> DirectiveContext<'a, S>
where
    S::Span: 'static,
{
    fn analyze(self, directive: Directive) -> TokenStreamSemantics<'a, S> {
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

    fn analyze_data(mut self, width: Width) -> TokenStreamSemantics<'a, S> {
        for arg in self.args {
            let expr = match self.session.expect_const(arg) {
                Ok(expr) => expr,
                Err(()) => return self.session,
            };
            self.session
                .session
                .emit_fragment(Fragment::Immediate(expr, width))
        }
        self.session
    }

    fn analyze_ds(mut self) -> TokenStreamSemantics<'a, S> {
        if let Some(arg) = single_arg(self.span, self.args, self.session.session) {
            let result = self.session.expect_const(arg);
            if let Ok(bytes) = result {
                self.session
                    .session
                    .emit_fragment(Fragment::Reserved(bytes))
            }
        }
        self.session
    }

    fn analyze_equ(mut self) -> TokenStreamSemantics<'a, S> {
        let (symbol, _) = self.label.take().unwrap();
        if let Some(arg) = single_arg(self.span, self.args, self.session.session) {
            self.session.define_symbol_with_params(symbol, arg);
        }
        self.session
    }

    fn analyze_section(mut self) -> TokenStreamSemantics<'a, S> {
        let (name, span) = self.label.take().unwrap().0;
        let id = self.session.reloc_lookup(name, span.clone());
        self.session.session.start_section(id, span);
        self.session
    }

    fn analyze_endc(self) -> TokenStreamSemantics<'a, S> {
        self.session
    }

    fn analyze_if(mut self) -> TokenStreamSemantics<'a, S> {
        match single_arg(self.span, self.args, self.session.session) {
            Some(arg) => {
                let value = self.session.expect_const(arg);
                match self.session.session.is_non_zero(value.unwrap()) {
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

    fn analyze_include(self) -> TokenStreamSemantics<'a, S> {
        let (path, span) = match reduce_include(self.span, self.args, self.session.session) {
            Some(result) => result,
            None => return self.session,
        };
        let result = self.session.session.analyze_file(path, Some(span.clone()));
        if let Err(error) = result {
            self.session
                .session
                .emit_diag(Message::CodebaseError { error }.at(span))
        }
        Semantics {
            session: self.session.session,
            state: TokenStreamState::new(),
        }
    }

    fn analyze_macro(self) -> TokenStreamSemantics<'a, S> {
        if self.label.is_none() {
            let span = self.span;
            self.session
                .session
                .emit_diag(Message::MacroRequiresName.at(span))
        }
        set_state!(
            self.session,
            TokenLineState {
                context: TokenContext::MacroDef(MacroDefState::new(self.label))
            }
            .into()
        )
    }

    fn analyze_org(mut self) -> TokenStreamSemantics<'a, S> {
        if let Some(arg) = single_arg(self.span, self.args, self.session.session) {
            let result = self.session.expect_const(arg);
            if let Ok(value) = result {
                self.session.session.set_origin(value)
            }
        }
        self.session
    }
}

fn reduce_include<D: Diagnostics<S>, S: Clone>(
    span: S,
    args: Vec<ParsedArg<S>>,
    diagnostics: &mut D,
) -> Option<(StringRef, S)> {
    let arg = match single_arg(span, args, diagnostics) {
        Some(arg) => arg,
        None => return None,
    };
    let result = match arg {
        ParsedArg::String(path, span) => Ok((path, span)),
        ParsedArg::Bare(expr) => Err(Some(expr.span())),
        ParsedArg::Parenthesized(_, span) => Err(Some(span)),
        ParsedArg::Error => Err(None),
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

    use crate::assembler::semantics::tests::Event;
    use crate::assembler::session::NameEntry;
    use crate::assembler::syntax::Literal;
    use crate::codebase::CodebaseError;
    use crate::expr::{Atom, Expr, ParamId};
    use crate::object::{Symbol, SymbolId, UserDefId};

    use std::borrow::Borrow;

    #[test]
    fn include_file() {
        let filename = "file.asm";
        let mut fixture = TestFixture::new();
        fixture.fs.add(filename, &[]);
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("INCLUDE".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            actions.act_on_atom(ExprAtom::Literal(Literal::String(filename.into())), ());
            actions.did_parse_arg().did_parse_instr().did_parse_line(());
        }
        assert_eq!(
            session.log(),
            [Event::AnalyzeFile {
                path: filename.into(),
                from: Some(())
            }]
        )
    }

    #[test]
    fn set_origin() {
        let origin = 0x3000;
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("ORG".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            actions.act_on_atom(mk_literal(origin), ());
            actions.did_parse_arg().did_parse_instr().did_parse_line(());
        }
        assert_eq!(
            session.log(),
            [Event::SetOrigin {
                addr: origin.into()
            }]
        )
    }

    #[test]
    fn emit_byte_items() {
        test_data_items_emission("DB", mk_byte, [0x42, 0x78])
    }

    #[test]
    fn emit_word_items() {
        test_data_items_emission("DW", mk_word, [0x4332, 0x780f])
    }

    fn mk_byte(byte: i32) -> Fragment<Expr<SymbolId, ()>> {
        Fragment::Immediate((byte).into(), Width::Byte)
    }

    fn mk_word(word: i32) -> Fragment<Expr<SymbolId, ()>> {
        Fragment::Immediate((word).into(), Width::Word)
    }

    fn test_data_items_emission(
        directive: &str,
        mk_item: impl Fn(i32) -> Fragment<Expr<SymbolId, ()>>,
        data: impl Borrow<[i32]>,
    ) {
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr(directive.into(), ())
                .into_builtin_instr();
            for datum in data.borrow().iter() {
                let mut arg = actions.will_parse_arg();
                arg.act_on_atom(mk_literal(*datum), ());
                actions = arg.did_parse_arg();
            }
            actions.did_parse_instr().did_parse_line(());
        }
        assert_eq!(
            session.log(),
            data.borrow()
                .iter()
                .cloned()
                .map(mk_item)
                .map(|fragment| Event::EmitFragment { fragment })
                .collect::<Vec<_>>()
        )
    }

    #[test]
    fn reserve_3_bytes() {
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("DS".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            actions.act_on_atom(mk_literal(3), ());
            actions.did_parse_arg().did_parse_instr().did_parse_line(());
        }
        assert_eq!(
            session.log(),
            [Event::EmitFragment {
                fragment: Fragment::Reserved(3.into())
            }]
        )
    }

    fn mk_literal(n: i32) -> ExprAtom<StringRef, Literal> {
        ExprAtom::Literal(Literal::Number(n))
    }

    #[test]
    fn ds_with_malformed_expr() {
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("DS".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            actions.act_on_atom(ExprAtom::Ident("A".into()), ());
            actions.did_parse_arg().did_parse_instr().did_parse_line(());
        }
        assert_eq!(
            session.log(),
            [Event::EmitDiag {
                diag: Message::KeywordInExpr { keyword: () }.at(()).into()
            }]
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
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("INCLUDE".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            actions.act_on_atom(mk_literal(7), ());
            actions.did_parse_arg().did_parse_instr().did_parse_line(());
        }
        assert_eq!(
            session.log(),
            [Event::EmitDiag {
                diag: Message::ExpectedString.at(()).into()
            }]
        )
    }

    #[test]
    fn data_with_malformed_expr() {
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("DB".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            actions.act_on_atom(ExprAtom::Ident("A".into()), ());
            actions.did_parse_arg().did_parse_instr().did_parse_line(());
        }
        assert_eq!(
            session.log(),
            [Event::EmitDiag {
                diag: Message::KeywordInExpr { keyword: () }.at(()).into()
            }]
        )
    }

    #[test]
    fn include_file_with_invalid_utf8() {
        let name = "invalid_utf8.s";
        let mut fixture = TestFixture::new();
        fixture.fs.add(name, &[0x5a, 0x0a, 0xf6, 0xa6]);
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            let mut context = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("INCLUDE".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            context.act_on_atom(ExprAtom::Literal(Literal::String(name.into())), ());
            context.did_parse_arg().did_parse_instr().did_parse_line(());
        }
        assert_eq!(
            session.log(),
            [
                Event::AnalyzeFile {
                    path: name.into(),
                    from: Some(())
                },
                Event::EmitDiag {
                    diag: Message::CodebaseError {
                        error: CodebaseError::Utf8Error
                    }
                    .at(())
                    .into()
                }
            ]
        )
    }

    #[test]
    fn include_nonexistent_file() {
        let name = "nonexistent.s";
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            let mut context = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("INCLUDE".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            context.act_on_atom(ExprAtom::Literal(Literal::String(name.into())), ());
            context.did_parse_arg().did_parse_instr().did_parse_line(());
        }
        assert_eq!(
            session.log(),
            [
                Event::AnalyzeFile {
                    path: name.into(),
                    from: Some(())
                },
                Event::EmitDiag {
                    diag: Message::CodebaseError {
                        error: CodebaseError::IoError("file does not exist".to_string())
                    }
                    .at(())
                    .into()
                }
            ]
        )
    }

    #[test]
    fn define_symbol() {
        let symbol = "sym";
        let value = 3;
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_label((symbol.into(), ()))
                .did_parse_label()
                .will_parse_instr("EQU".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            actions.act_on_atom(ExprAtom::Literal(Literal::Number(value)), ());
            actions.did_parse_arg().did_parse_instr().did_parse_line(());
        }
        assert_eq!(
            session.log(),
            [
                Event::DefineNameWithVisibility {
                    ident: symbol.into(),
                    visibility: Visibility::Global,
                    entry: NameEntry::Symbol(Symbol::UserDef(UserDefId(0)))
                },
                Event::DefineSymbol {
                    name: Symbol::UserDef(UserDefId(0)),
                    span: (),
                    expr: value.into()
                }
            ]
        )
    }

    #[test]
    fn define_fn_with_param() {
        let name = "my_fn";
        let param = "param";
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            let mut label_actions = actions
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
                .act_on_eos(());
        }
        assert_eq!(
            session.log(),
            [
                Event::DefineNameWithVisibility {
                    ident: name.into(),
                    visibility: Visibility::Global,
                    entry: NameEntry::Symbol(Symbol::UserDef(UserDefId(0)))
                },
                Event::DefineSymbol {
                    name: Symbol::UserDef(UserDefId(0)),
                    span: (),
                    expr: Atom::from(ParamId(0)).into()
                },
            ]
        )
    }

    #[test]
    fn start_section() {
        let name = "hot_stuff";
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_label((name.into(), ()))
                .did_parse_label()
                .will_parse_instr("SECTION".into(), ())
                .into_builtin_instr()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(());
        }
        assert_eq!(
            session.log(),
            [
                Event::DefineNameWithVisibility {
                    ident: name.into(),
                    visibility: Visibility::Global,
                    entry: NameEntry::Symbol(Symbol::UserDef(UserDefId(0)))
                },
                Event::StartSection {
                    name: Symbol::UserDef(UserDefId(0)),
                    span: ()
                }
            ]
        )
    }

    #[test]
    fn taken_if_remains_in_instr_mode() {
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        let session = analyze_directive(
            (Directive::If, ()),
            None,
            vec![ParsedArg::Bare(Expr::from_atom(1.into(), ()))],
            session.semantic_actions(),
        );
        assert_eq!(
            session.state,
            TokenStreamState {
                mode: LineRule::InstrLine(InstrLineState { label: None })
            }
        );
    }

    #[test]
    fn ignore_instrs_in_untaken_if() {
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        let session = analyze_directive(
            (Directive::If, ()),
            None,
            vec![ParsedArg::Bare(Expr::from_atom(0.into(), ()))],
            session.semantic_actions(),
        );
        assert_eq!(
            session.state,
            TokenStreamState {
                mode: LineRule::TokenLine(TokenLineState {
                    context: TokenContext::FalseIf
                })
            }
        );
    }

    fn test_unary_directive_without_args(directive: &str) {
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        {
            let actions = session.semantic_actions();
            actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr(directive.into(), ())
                .into_builtin_instr()
                .did_parse_instr()
                .did_parse_line(());
        }
        assert_eq!(
            session.log(),
            [Event::EmitDiag {
                diag: Message::OperandCount {
                    actual: 0,
                    expected: 1
                }
                .at(())
                .into()
            }]
        )
    }
}
