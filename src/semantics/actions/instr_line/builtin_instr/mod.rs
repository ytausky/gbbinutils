use super::*;

use crate::semantics::actions::TokenStreamState;
use crate::semantics::arg::*;
use crate::semantics::keywords::{Directive, Mnemonic};
use crate::session::builder::SymbolSource;
use crate::session::resolve::NameTable;
use crate::syntax::actions::{BuiltinInstrContext, InstrFinalizer};

mod arg;
mod cpu_instr;
mod directive;

impl<S: Session> From<BuiltinInstrState<S>>
    for TokenStreamState<
        <S as IdentSource>::Ident,
        <S as StringSource>::StringRef,
        <S as SpanSource>::Span,
    >
{
    fn from(_: BuiltinInstrState<S>) -> Self {
        InstrLineState::new().into()
    }
}

impl<'a, S: Session> BuiltinInstrContext for BuiltinInstrSemantics<'a, S>
where
    S::Ident: 'static,
    S::StringRef: 'static,
    S::Span: 'static,
{
    type ArgContext = ArgSemantics<'a, S>;

    fn will_parse_arg(self) -> Self::ArgContext {
        Semantics {
            session: self.session,
            state: ExprBuilder::new(self.state),
            tokens: self.tokens,
        }
    }
}

impl<'a, S: Session> InstrFinalizer for BuiltinInstrSemantics<'a, S>
where
    S::Ident: 'static,
    S::StringRef: 'static,
    S::Span: 'static,
{
    type Next = TokenStreamSemantics<'a, S>;

    fn did_parse_instr(self) -> Self::Next {
        let args = self.state.args;
        let mut session = set_state!(self, InstrLineState::new().into());
        match self.state.mnemonic.item {
            BuiltinMnemonic::CpuInstr(cpu_instr) => {
                analyze_mnemonic(
                    (&cpu_instr, self.state.mnemonic.span),
                    args,
                    &mut session.session,
                );
                session.map_state(Into::into)
            }
            BuiltinMnemonic::Directive(directive) => directive::analyze_directive(
                (directive, self.state.mnemonic.span),
                self.state.label,
                args,
                session,
            ),
        }
    }
}

impl<'a, S: Session, T> Semantics<'a, S, T, S::Ident, S::StringRef, S::Span> {
    fn expect_const(
        &mut self,
        arg: ParsedArg<S::Ident, S::StringRef, S::Span>,
    ) -> Result<Expr<S::SymbolId, S::Span>, ()> {
        match self.session.resolve_names(arg)? {
            Arg::Bare(DerefableArg::Const(value)) => Ok(value),
            Arg::Bare(DerefableArg::Symbol(_, span)) => {
                let keyword = self.session.strip_span(&span);
                self.session
                    .emit_diag(Message::KeywordInExpr { keyword }.at(span));
                Err(())
            }
            Arg::Error => Err(()),
            _ => unimplemented!(),
        }
    }

    fn define_symbol_with_params(
        &mut self,
        (name, span): (S::Ident, S::Span),
        expr: ParsedArg<S::Ident, S::StringRef, S::Span>,
    ) {
        if let Ok(expr) = self.expect_const(expr) {
            let id = self.reloc_lookup(name, span.clone());
            self.session.define_symbol(id, span, expr);
        }
    }
}

impl From<Directive> for BuiltinMnemonic {
    fn from(directive: Directive) -> Self {
        BuiltinMnemonic::Directive(directive)
    }
}

impl From<Mnemonic> for BuiltinMnemonic {
    fn from(mnemonic: Mnemonic) -> Self {
        BuiltinMnemonic::CpuInstr(mnemonic)
    }
}

fn analyze_mnemonic<S: Session>(
    name: (&Mnemonic, S::Span),
    args: BuiltinInstrArgs<S::Ident, S::StringRef, S::Span>,
    session: &mut S,
) {
    let mut operands = Vec::new();
    let mut error = false;
    for arg in args {
        if let Ok(arg) = session.resolve_names(arg) {
            let operand = cpu_instr::operand::analyze_operand(arg, name.0.context(), session);
            operands.push(operand)
        } else {
            error = true;
        }
    }
    if !error {
        cpu_instr::analyze_instruction(name, operands, session)
    }
}

trait Resolve<I, R, S>: SymbolSource {
    fn resolve_names(
        &mut self,
        arg: ParsedArg<I, R, S>,
    ) -> Result<Arg<Expr<Self::SymbolId, S>, R, S>, ()>;
}

trait ClassifyExpr<I, S>: SymbolSource {
    fn classify_expr(
        &mut self,
        expr: Expr<I, S>,
    ) -> Result<SymbolOrResolvedExpr<Self::SymbolId, S>, ()>;
}

enum SymbolOrResolvedExpr<Sym, S> {
    Symbol(OperandSymbol, S),
    Expr(Expr<Sym, S>),
}

impl<T, I, R, S> Resolve<I, R, S> for T
where
    T: NameTable<I, Keyword = &'static Keyword> + Diagnostics<S> + AllocSymbol<S>,
    S: Clone,
{
    fn resolve_names(
        &mut self,
        arg: ParsedArg<I, R, S>,
    ) -> Result<Arg<Expr<Self::SymbolId, S>, R, S>, ()> {
        match arg {
            ParsedArg::Bare(expr) => match self.classify_expr(expr)? {
                SymbolOrResolvedExpr::Symbol(symbol, span) => {
                    Ok(Arg::Bare(DerefableArg::Symbol(symbol, span)))
                }
                SymbolOrResolvedExpr::Expr(expr) => Ok(Arg::Bare(DerefableArg::Const(expr))),
            },
            ParsedArg::Parenthesized(expr, span) => match self.classify_expr(expr)? {
                SymbolOrResolvedExpr::Symbol(symbol, inner_span) => {
                    Ok(Arg::Deref(DerefableArg::Symbol(symbol, inner_span), span))
                }
                SymbolOrResolvedExpr::Expr(expr) => Ok(Arg::Deref(DerefableArg::Const(expr), span)),
            },
            ParsedArg::String(string, span) => Ok(Arg::String(string, span)),
            ParsedArg::Error => Ok(Arg::Error),
        }
    }
}

impl<T, I, S> ClassifyExpr<I, S> for T
where
    T: NameTable<I, Keyword = &'static Keyword> + Diagnostics<S> + AllocSymbol<S>,
    S: Clone,
{
    fn classify_expr(
        &mut self,
        mut expr: Expr<I, S>,
    ) -> Result<SymbolOrResolvedExpr<Self::SymbolId, S>, ()> {
        if expr.0.len() == 1 {
            let node = expr.0.pop().unwrap();
            match node.item {
                ExprOp::Atom(Atom::Name(name)) => {
                    match self.resolve_name(&name) {
                        Some(ResolvedName::Keyword(Keyword::Operand(operand))) => {
                            Ok(SymbolOrResolvedExpr::Symbol(*operand, node.span))
                        }
                        Some(ResolvedName::Keyword(_)) => {
                            let keyword = self.strip_span(&node.span);
                            self.emit_diag(Message::KeywordInExpr { keyword }.at(node.span));
                            Err(())
                        }
                        Some(ResolvedName::Symbol(id)) => Ok(SymbolOrResolvedExpr::Expr(Expr(
                            vec![ExprOp::Atom(Atom::Name(id)).with_span(node.span)],
                        ))),
                        None => {
                            let id = self.alloc_symbol(node.span.clone());
                            self.define_name(name, ResolvedName::Symbol(id.clone()));
                            Ok(SymbolOrResolvedExpr::Expr(Expr(vec![ExprOp::Atom(
                                Atom::Name(id),
                            )
                            .with_span(node.span)])))
                        }
                        Some(ResolvedName::Macro(_)) => todo!(),
                    }
                }
                ExprOp::Atom(Atom::Const(n)) => {
                    Ok(SymbolOrResolvedExpr::Expr(Expr(vec![ExprOp::Atom(
                        Atom::Const(n),
                    )
                    .with_span(node.span)])))
                }
                ExprOp::Atom(Atom::Location) => {
                    Ok(SymbolOrResolvedExpr::Expr(Expr(vec![ExprOp::Atom(
                        Atom::Location,
                    )
                    .with_span(node.span)])))
                }
                ExprOp::Atom(Atom::Param(id)) => {
                    Ok(SymbolOrResolvedExpr::Expr(Expr(vec![ExprOp::Atom(
                        Atom::Param(id),
                    )
                    .with_span(node.span)])))
                }
                _ => panic!("first node in expression must be an atom"),
            }
        } else {
            let mut nodes = Vec::new();
            let mut error = false;
            for node in expr.0 {
                match node.item {
                    ExprOp::Atom(Atom::Name(name)) => match self.resolve_name(&name) {
                        Some(ResolvedName::Keyword(_)) => {
                            let keyword = self.strip_span(&node.span);
                            self.emit_diag(Message::KeywordInExpr { keyword }.at(node.span));
                            error = true
                        }
                        Some(ResolvedName::Symbol(id)) => {
                            nodes.push(ExprOp::Atom(Atom::Name(id)).with_span(node.span))
                        }
                        None => {
                            let id = self.alloc_symbol(node.span.clone());
                            self.define_name(name, ResolvedName::Symbol(id.clone()));
                            nodes.push(ExprOp::Atom(Atom::Name(id)).with_span(node.span))
                        }
                        Some(ResolvedName::Macro(_)) => todo!(),
                    },
                    ExprOp::Atom(Atom::Const(n)) => {
                        nodes.push(ExprOp::Atom(Atom::Const(n)).with_span(node.span))
                    }
                    ExprOp::Atom(Atom::Location) => {
                        nodes.push(ExprOp::Atom(Atom::Location).with_span(node.span))
                    }
                    ExprOp::Atom(Atom::Param(id)) => {
                        nodes.push(ExprOp::Atom(Atom::Param(id)).with_span(node.span))
                    }
                    ExprOp::Binary(op) => nodes.push(ExprOp::Binary(op).with_span(node.span)),
                    ExprOp::FnCall(arity) => nodes.push(ExprOp::FnCall(arity).with_span(node.span)),
                }
            }
            if !error {
                Ok(SymbolOrResolvedExpr::Expr(Expr(nodes)))
            } else {
                Err(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::analyze::Literal::*;
    use crate::diag::{DiagnosticsEvent, Message, MockSpan};
    use crate::semantics::actions::tests::collect_semantic_actions;
    use crate::syntax::actions::*;
    use crate::syntax::actions::{ExprAtom::*, Operator::*};

    #[ignore]
    #[test]
    fn diagnose_literal_as_fn_name() {
        assert_eq!(
            collect_semantic_actions::<_, MockSpan<_>>(|actions| {
                let mut actions = actions
                    .will_parse_line()
                    .into_instr_line()
                    .will_parse_instr("DB".into(), "db".into())
                    .into_builtin_instr()
                    .will_parse_arg();
                actions.act_on_atom(Literal(Number(7)), "literal".into());
                actions.act_on_operator(FnCall(0), "call".into());
                actions
                    .did_parse_arg()
                    .did_parse_instr()
                    .did_parse_line("eol".into())
                    .act_on_eos("eos".into())
            }),
            [DiagnosticsEvent::EmitDiag(
                Message::OnlyIdentsCanBeCalled.at("literal".into()).into()
            )
            .into()]
        );
    }
}
