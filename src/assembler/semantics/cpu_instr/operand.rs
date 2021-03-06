use super::{Condition, PtrReg, Reg16, RegPair, M};

use crate::assembler::semantics::*;
use crate::diagnostics::*;
use crate::expr::Expr;
use crate::span::{Source, SpanSource};

#[derive(Debug, PartialEq)]
pub(in super::super) enum Operand<S> {
    Atom(AtomKind, S),
    Const(Expr<Name, S>),
    Deref(Expr<Name, S>),
}

impl<S: Clone> SpanSource for Operand<S> {
    type Span = S;
}

impl<S: Clone> Source for Operand<S> {
    fn span(&self) -> Self::Span {
        match self {
            Operand::Atom(_, span) => (*span).clone(),
            Operand::Const(expr) | Operand::Deref(expr) => expr.span(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in super::super) enum AtomKind {
    Simple(M),
    Condition(Condition),
    Reg16(Reg16),
    RegPair(RegPair),
    DerefPtrReg(PtrReg),
    DerefC,
}

#[derive(Clone, Copy)]
pub enum Context {
    Branch,
    Stack,
    Other,
}

pub(in super::super) fn analyze_operand<D, S>(
    expr: Arg<S>,
    context: Context,
    diagnostics: &mut D,
) -> Result<Operand<S>, ()>
where
    D: Diagnostics<S>,
    S: Clone,
{
    match expr {
        Arg::Bare(BareArg::Const(value)) => Ok(Operand::Const(value)),
        Arg::Bare(BareArg::OperandKeyword(symbol, span)) => {
            analyze_keyword_operand((symbol, span), context, diagnostics)
        }
        Arg::Deref(BareArg::Const(value), _) => Ok(Operand::Deref(value)),
        Arg::Deref(BareArg::OperandKeyword(symbol, inner_span), outer_span) => {
            analyze_deref_operand_keyword((symbol, &inner_span), outer_span, diagnostics)
        }
        Arg::String(_, span) => {
            diagnostics.emit_diag(Message::StringInInstruction.at(span));
            Err(())
        }
        Arg::Error => Err(()),
    }
}

fn analyze_deref_operand_keyword<D, S>(
    keyword: (OperandKeyword, &S),
    deref: S,
    diagnostics: &mut D,
) -> Result<Operand<S>, ()>
where
    D: Diagnostics<S>,
{
    match try_deref_operand_keyword(keyword.0) {
        Ok(atom) => Ok(Operand::Atom(atom, deref)),
        Err(category) => {
            let stripped = diagnostics.strip_span(keyword.1);
            diagnostics.emit_diag(
                Message::CannotDereference {
                    category,
                    operand: stripped,
                }
                .at(deref),
            );
            Err(())
        }
    }
}

fn try_deref_operand_keyword(symbol: OperandKeyword) -> Result<AtomKind, KeywordOperandCategory> {
    use self::OperandKeyword::*;

    match symbol {
        Bc => Ok(AtomKind::DerefPtrReg(PtrReg::Bc)),
        C => Ok(AtomKind::DerefC),
        De => Ok(AtomKind::DerefPtrReg(PtrReg::De)),
        Hl => Ok(AtomKind::Simple(M::DerefHl)),
        Hld => Ok(AtomKind::DerefPtrReg(PtrReg::Hld)),
        Hli => Ok(AtomKind::DerefPtrReg(PtrReg::Hli)),
        A | B | D | E | H | L | Sp => Err(KeywordOperandCategory::Reg),
        Af => Err(KeywordOperandCategory::RegPair),
        Nc | Nz | Z => Err(KeywordOperandCategory::ConditionCode),
    }
}

fn analyze_keyword_operand<D, S>(
    (symbol, span): (OperandKeyword, S),
    context: Context,
    diagnostics: &mut D,
) -> Result<Operand<S>, ()>
where
    D: Diagnostics<S>,
    S: Clone,
{
    use self::Context::*;
    use self::OperandKeyword::*;

    let kind = match symbol {
        A => AtomKind::Simple(M::A),
        Af => AtomKind::RegPair(RegPair::Af),
        B => AtomKind::Simple(M::B),
        Bc => match context {
            Stack => AtomKind::RegPair(RegPair::Bc),
            _ => AtomKind::Reg16(Reg16::Bc),
        },
        C => match context {
            Branch => AtomKind::Condition(Condition::C),
            _ => AtomKind::Simple(M::C),
        },
        D => AtomKind::Simple(M::D),
        De => match context {
            Stack => AtomKind::RegPair(RegPair::De),
            _ => AtomKind::Reg16(Reg16::De),
        },
        E => AtomKind::Simple(M::E),
        H => AtomKind::Simple(M::H),
        Hl => match context {
            Stack => AtomKind::RegPair(RegPair::Hl),
            _ => AtomKind::Reg16(Reg16::Hl),
        },
        Hld | Hli => {
            let stripped = diagnostics.strip_span(&span);
            diagnostics.emit_diag(Message::MustBeDeref { operand: stripped }.at(span));
            return Err(());
        }
        L => AtomKind::Simple(M::L),
        Nc => AtomKind::Condition(Condition::Nc),
        Nz => AtomKind::Condition(Condition::Nz),
        Sp => AtomKind::Reg16(Reg16::Sp),
        Z => AtomKind::Condition(Condition::Z),
    };
    Ok(Operand::Atom(kind, span))
}

pub struct OperandCounter<I> {
    operands: I,
    count: usize,
}

impl<I> OperandCounter<I> {
    pub fn new(operands: I) -> OperandCounter<I> {
        OperandCounter { operands, count: 0 }
    }

    pub fn seen(&self) -> usize {
        self.count
    }
}

impl<I: Iterator> Iterator for OperandCounter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.operands.next();
        if item.is_some() {
            self.count += 1;
        }
        item
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    use crate::assembler::session::MacroId;
    use crate::object::Name;

    use std::fmt::Debug;

    #[test]
    fn analyze_deref_bc() {
        analyze_deref_ptr_reg(PtrReg::Bc)
    }

    #[test]
    fn analyze_deref_de() {
        analyze_deref_ptr_reg(PtrReg::De)
    }

    #[test]
    fn analyze_deref_hli() {
        analyze_deref_ptr_reg(PtrReg::Hli)
    }

    #[test]
    fn analyze_deref_hld() {
        analyze_deref_ptr_reg(PtrReg::Hld)
    }

    fn analyze_deref_ptr_reg(ptr_reg: PtrReg) {
        let expr = Arg::Deref(BareArg::OperandKeyword(ptr_reg.into(), 0.into()), 1.into());
        assert_eq!(
            analyze_operand(expr, Context::Other),
            Ok(Operand::Atom(AtomKind::DerefPtrReg(ptr_reg), 1.into()))
        )
    }

    pub(in crate::assembler::semantics::cpu_instr) type Event<S> =
        crate::assembler::session::Event<Name, MacroId, S, S>;

    type OperandResult<S> = Result<Operand<S>, Vec<Event<S>>>;

    fn analyze_operand<S: Clone + Debug>(
        expr: Arg<MockSpan<S>>,
        context: Context,
    ) -> OperandResult<MockSpan<S>> {
        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        let result = super::analyze_operand(expr, context, &mut session);
        result.map_err(|_| session.log().to_vec())
    }

    #[test]
    fn analyze_deref_af() {
        let parsed_expr = Arg::Deref(
            BareArg::OperandKeyword(OperandKeyword::Af, 0.into()),
            1.into(),
        );
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Err(vec![Event::EmitDiag {
                diag: CompactDiag::from(
                    Message::CannotDereference {
                        category: KeywordOperandCategory::RegPair,
                        operand: 0.into(),
                    }
                    .at(1.into())
                )
            }])
        )
    }

    #[test]
    fn analyze_string_in_instruction() {
        let span = 0;
        let parsed_expr = Arg::String("some_string".into(), span.into());
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Err(vec![Event::EmitDiag {
                diag: CompactDiag::from(Message::StringInInstruction.at(span.into()))
            }])
        )
    }

    #[test]
    fn analyze_bare_hld() {
        test_bare_ptr_reg(OperandKeyword::Hld)
    }

    #[test]
    fn analyze_bare_hli() {
        test_bare_ptr_reg(OperandKeyword::Hli)
    }

    fn test_bare_ptr_reg(symbol: OperandKeyword) {
        let span = MockSpan::from(0);
        let expr = Arg::Bare(BareArg::OperandKeyword(symbol, span.clone()));
        assert_eq!(
            analyze_operand(expr, Context::Other),
            Err(vec![Event::EmitDiag {
                diag: CompactDiag::from(
                    Message::MustBeDeref {
                        operand: span.clone()
                    }
                    .at(span)
                )
            }])
        )
    }
}
