pub(super) use self::directive::Directive;
pub(super) use self::mnemonic::{Mnemonic, *};

use self::args::*;
use self::operand::OperandSymbol;

use super::*;

use crate::analysis::backend::{Finish, LocationCounter, Name, PushOp};
use crate::analysis::session::Session;
use crate::diag::span::{MergeSpans, StripSpan};
use crate::diag::{CompactDiag, Diagnostics, EmitDiag, Message};
use crate::model::{BinOp, FnCall, Item};

mod args;
mod directive;
mod mnemonic;
mod operand;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analysis) enum BuiltinInstr {
    Directive(Directive),
    Mnemonic(Mnemonic),
}

impl From<Directive> for BuiltinInstr {
    fn from(directive: Directive) -> Self {
        BuiltinInstr::Directive(directive)
    }
}

impl From<Mnemonic> for BuiltinInstr {
    fn from(mnemonic: Mnemonic) -> Self {
        BuiltinInstr::Mnemonic(mnemonic)
    }
}

pub(in crate::analysis) struct BuiltinInstrActions<S: Session> {
    parent: SemanticActions<S>,
    command: (BuiltinInstr, S::Span),
    args: BuiltinInstrArgs<S::Ident, S::StringRef, S::Span>,
    has_errors: bool,
}

impl<S: Session> BuiltinInstrActions<S> {
    pub(super) fn new(
        parent: SemanticActions<S>,
        command: (BuiltinInstr, S::Span),
    ) -> BuiltinInstrActions<S> {
        BuiltinInstrActions {
            parent,
            command,
            args: Vec::new(),
            has_errors: false,
        }
    }
}

impl<S: Session> MergeSpans<S::Span> for BuiltinInstrActions<S> {
    fn merge_spans(&mut self, left: &S::Span, right: &S::Span) -> S::Span {
        self.parent.merge_spans(left, right)
    }
}

impl<S: Session> StripSpan<S::Span> for BuiltinInstrActions<S> {
    type Stripped = S::Stripped;

    fn strip_span(&mut self, span: &S::Span) -> Self::Stripped {
        self.parent.strip_span(span)
    }
}

impl<S: Session> EmitDiag<S::Span, S::Stripped> for BuiltinInstrActions<S> {
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<S::Span, S::Stripped>>) {
        self.has_errors = true;
        self.parent.emit_diag(diag)
    }
}

impl<S: Session> BuiltinInstrContext<S::Span> for BuiltinInstrActions<S>
where
    S::Ident: AsRef<str>,
{
    type Ident = S::Ident;
    type Literal = Literal<S::StringRef>;
    type ArgContext = ExprBuilder<S::Ident, S::StringRef, S::Span, Self>;
    type Parent = SemanticActions<S>;

    fn add_argument(self) -> Self::ArgContext {
        ExprBuilder {
            stack: Vec::new(),
            parent: self,
        }
    }

    fn exit(mut self) -> Self::Parent {
        if !self.has_errors {
            let prepared = PreparedBuiltinInstr::new(self.command, &mut self.parent);
            self.parent.define_label_if_present();
            prepared.exec(self.args, &mut self.parent)
        }
        self.parent
    }
}

enum PreparedBuiltinInstr<S: Session> {
    Binding((Directive, S::Span), Option<Label<S::Ident, S::Span>>),
    Directive((Directive, S::Span)),
    Mnemonic((Mnemonic, S::Span)),
}

impl<S: Session> PreparedBuiltinInstr<S> {
    fn new((command, span): (BuiltinInstr, S::Span), stmt: &mut SemanticActions<S>) -> Self {
        match command {
            BuiltinInstr::Directive(directive) if directive.requires_symbol() => {
                PreparedBuiltinInstr::Binding((directive, span), stmt.label.take())
            }
            BuiltinInstr::Directive(directive) => {
                PreparedBuiltinInstr::Directive((directive, span))
            }
            BuiltinInstr::Mnemonic(mnemonic) => PreparedBuiltinInstr::Mnemonic((mnemonic, span)),
        }
    }

    fn exec(
        self,
        args: BuiltinInstrArgs<S::Ident, S::StringRef, S::Span>,
        actions: &mut SemanticActions<S>,
    ) {
        match self {
            PreparedBuiltinInstr::Binding(binding, label) => {
                directive::analyze_directive(binding, label, args, actions)
            }
            PreparedBuiltinInstr::Directive(directive) => {
                directive::analyze_directive(directive, None, args, actions)
            }
            PreparedBuiltinInstr::Mnemonic(mnemonic) => analyze_mnemonic(mnemonic, args, actions),
        }
    }
}

impl Directive {
    fn requires_symbol(self) -> bool {
        match self {
            Directive::Equ | Directive::Section => true,
            _ => false,
        }
    }
}

pub(crate) struct ExprBuilder<I, R, S, P> {
    stack: Vec<Arg<I, R, S>>,
    parent: P,
}

impl<I, R, S, P> ExprBuilder<I, R, S, P> {
    fn pop(&mut self) -> Arg<I, R, S> {
        self.stack.pop().unwrap_or_else(|| unreachable!())
    }
}

delegate_diagnostics! {
    {I, R, S, P: Diagnostics<S>}, ExprBuilder<I, R, S, P>, {parent}, P, S
}

impl<S: Session> FinalContext
    for ExprBuilder<S::Ident, S::StringRef, S::Span, BuiltinInstrActions<S>>
{
    type ReturnTo = BuiltinInstrActions<S>;

    fn exit(mut self) -> Self::ReturnTo {
        if !self.parent.has_errors {
            assert_eq!(self.stack.len(), 1);
            self.parent.args.push(self.stack.pop().unwrap());
        }
        self.parent
    }
}

impl<I, R, S, P> ExprContext<S> for ExprBuilder<I, R, S, P>
where
    I: AsRef<str>,
    S: Clone,
    Self: Diagnostics<S>,
{
    type Ident = I;
    type Literal = Literal<R>;

    fn push_atom(&mut self, atom: (ExprAtom<Self::Ident, Self::Literal>, S)) {
        self.stack.push(Arg {
            variant: ArgVariant::Atom(match atom.0 {
                ExprAtom::Ident(ident) => OPERAND_SYMBOLS
                    .iter()
                    .find(|(spelling, _)| spelling.eq_ignore_ascii_case(ident.as_ref()))
                    .map(|(_, symbol)| ArgAtom::OperandSymbol(*symbol))
                    .unwrap_or_else(|| ArgAtom::Ident(ident)),
                ExprAtom::Literal(literal) => ArgAtom::Literal(literal),
                ExprAtom::LocationCounter => ArgAtom::LocationCounter,
            }),
            span: atom.1,
        })
    }

    fn apply_operator(&mut self, (op, span): (Operator, S)) {
        let variant = match op {
            Operator::Unary(UnaryOperator::Parentheses) => {
                let inner = self.pop();
                ArgVariant::Unary(ArgUnaryOp::Parentheses, Box::new(inner))
            }
            Operator::Binary(binary) => {
                let rhs = self.pop();
                let lhs = self.pop();
                ArgVariant::Binary(binary, Box::new(lhs), Box::new(rhs))
            }
            Operator::FnCall(n) => {
                let args = self.stack.split_off(self.stack.len() - n);
                let name = self.pop();
                let name = (
                    match name.variant {
                        ArgVariant::Atom(ArgAtom::Ident(ident)) => ident,
                        _ => {
                            self.emit_diag(Message::OnlyIdentsCanBeCalled.at(name.span));
                            return;
                        }
                    },
                    name.span,
                );
                ArgVariant::FnCall(name, args)
            }
        };
        self.stack.push(Arg { variant, span })
    }
}

const OPERAND_SYMBOLS: &[(&str, OperandSymbol)] = &[
    ("a", OperandSymbol::A),
    ("af", OperandSymbol::Af),
    ("b", OperandSymbol::B),
    ("bc", OperandSymbol::Bc),
    ("c", OperandSymbol::C),
    ("d", OperandSymbol::D),
    ("de", OperandSymbol::De),
    ("e", OperandSymbol::E),
    ("h", OperandSymbol::H),
    ("hl", OperandSymbol::Hl),
    ("hld", OperandSymbol::Hld),
    ("hli", OperandSymbol::Hli),
    ("l", OperandSymbol::L),
    ("nc", OperandSymbol::Nc),
    ("nz", OperandSymbol::Nz),
    ("sp", OperandSymbol::Sp),
    ("z", OperandSymbol::Z),
];

fn analyze_mnemonic<S: Session>(
    name: (Mnemonic, S::Span),
    args: BuiltinInstrArgs<S::Ident, S::StringRef, S::Span>,
    actions: &mut SemanticActions<S>,
) {
    let operands: Vec<_> = args
        .into_iter()
        .map(|arg| {
            actions.build_value(&Default::default(), |builder| {
                operand::analyze_operand(arg, name.0.context(), builder)
            })
        })
        .collect();
    if let Ok(instruction) = mnemonic::analyze_instruction(name, operands, actions) {
        actions.session().emit_item(Item::Instruction(instruction))
    }
}

impl<S: Session> SemanticActions<S> {
    fn analyze_expr(
        &mut self,
        params: &Params<S::Ident, S::Span>,
        expr: Arg<S::Ident, S::StringRef, S::Span>,
    ) -> Result<S::Value, ()> {
        self.build_value(params, |mut builder| {
            let result = builder.eval_arg(expr);
            let (session, value) = builder.finish();
            (session, result.map(|()| value))
        })
    }

    fn define_symbol(
        &mut self,
        (name, span): (S::Ident, S::Span),
        params: &Params<S::Ident, S::Span>,
        expr: Arg<S::Ident, S::StringRef, S::Span>,
    ) -> Result<(), ()> {
        let mut result = Ok(());
        self.with_session(|mut session| {
            let id = session.reloc_lookup(name, span.clone());
            let mut builder = session
                .define_symbol(id, span)
                .resolve_names()
                .with_params(params);
            result = builder.eval_arg(expr);
            builder.finish()
        });
        result
    }
}

trait EvalArg<I, R, S: Clone> {
    fn eval_arg(&mut self, arg: Arg<I, R, S>) -> Result<(), ()>;
}

trait ArgEvaluator<N, S: Clone>:
    PushOp<LocationCounter, S>
    + PushOp<i32, S>
    + PushOp<Name<N>, S>
    + PushOp<BinOp, S>
    + PushOp<FnCall, S>
    + Diagnostics<S>
{
}

impl<T, N, S: Clone> ArgEvaluator<N, S> for T where
    Self: PushOp<LocationCounter, S>
        + PushOp<i32, S>
        + PushOp<Name<N>, S>
        + PushOp<BinOp, S>
        + PushOp<FnCall, S>
        + Diagnostics<S>
{
}

impl<'a, T, I, R, S> EvalArg<I, R, S> for T
where
    T: ArgEvaluator<I, S>,
    R: Eq,
    S: Clone,
{
    fn eval_arg(&mut self, arg: Arg<I, R, S>) -> Result<(), ()> {
        match arg.variant {
            ArgVariant::Atom(ArgAtom::Ident(ident)) => {
                self.push_op(Name(ident), arg.span);
                Ok(())
            }
            ArgVariant::Atom(ArgAtom::Literal(Literal::Number(n))) => {
                self.push_op(n, arg.span);
                Ok(())
            }
            ArgVariant::Atom(ArgAtom::OperandSymbol(_)) => Err(Message::KeywordInExpr {
                keyword: self.strip_span(&arg.span),
            }
            .at(arg.span)),
            ArgVariant::Atom(ArgAtom::Literal(Literal::String(_))) => {
                Err(Message::StringInInstruction.at(arg.span))
            }
            ArgVariant::Atom(ArgAtom::LocationCounter) => {
                self.push_op(LocationCounter, arg.span);
                Ok(())
            }
            ArgVariant::Unary(ArgUnaryOp::Parentheses, expr) => Ok(self.eval_arg(*expr)?),
            ArgVariant::Binary(binary, left, right) => {
                self.eval_arg(*left)?;
                self.eval_arg(*right)?;
                self.push_op(binary, arg.span);
                Ok(())
            }
            ArgVariant::FnCall((name, span), args) => {
                let n = args.len();
                for arg in args {
                    self.eval_arg(arg)?;
                }
                self.push_op(Name(name), span.clone());
                self.push_op(FnCall(n), span);
                Ok(())
            }
        }
        .map_err(|diagnostic| {
            self.emit_diag(diagnostic);
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::analysis::semantics::tests::collect_semantic_actions;
    use crate::analysis::syntax::*;
    use crate::analysis::syntax::{ExprAtom::*, Operator::*};
    use crate::analysis::Literal::*;
    use crate::diag::{DiagnosticsEvent, Message, MockSpan};

    #[test]
    fn diagnose_literal_as_fn_name() {
        assert_eq!(
            collect_semantic_actions::<_, MockSpan<_>>(|actions| {
                let mut actions = actions
                    .key_lookup("DB".into(), "db".into())
                    .command()
                    .unwrap()
                    .add_argument();
                actions.push_atom((Literal(Number(7)), "literal".into()));
                actions.apply_operator((FnCall(0), "call".into()));
                actions.exit().exit().exit()
            }),
            [DiagnosticsEvent::EmitDiag(
                Message::OnlyIdentsCanBeCalled.at("literal".into()).into()
            )
            .into()]
        );
    }
}
