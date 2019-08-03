use super::branch::*;

use crate::model::*;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analysis) enum Mnemonic {
    Alu(AluOperation),
    Bit(BitOperation),
    Branch(BranchKind),
    IncDec(IncDec),
    Ld,
    Ldhl,
    Misc(MiscOperation),
    Nullary(Nullary),
    Rst,
    Stack(StackOperation),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::analysis) enum StackOperation {
    Push,
    Pop,
}

pub(in crate::analysis::semantics) const ADC: Mnemonic = Mnemonic::Alu(AluOperation::Adc);
pub(in crate::analysis::semantics) const ADD: Mnemonic = Mnemonic::Alu(AluOperation::Add);
pub(in crate::analysis::semantics) const AND: Mnemonic = Mnemonic::Alu(AluOperation::And);
pub(in crate::analysis::semantics) const BIT: Mnemonic = Mnemonic::Bit(BitOperation::Bit);
pub(in crate::analysis::semantics) const CALL: Mnemonic =
    Mnemonic::Branch(BranchKind::Explicit(ExplicitBranch::Call));
pub(in crate::analysis::semantics) const CP: Mnemonic = Mnemonic::Alu(AluOperation::Cp);
pub(in crate::analysis::semantics) const CPL: Mnemonic = Mnemonic::Nullary(Nullary::Cpl);
pub(in crate::analysis::semantics) const DAA: Mnemonic = Mnemonic::Nullary(Nullary::Daa);
pub(in crate::analysis::semantics) const DEC: Mnemonic = Mnemonic::IncDec(IncDec::Dec);
pub(in crate::analysis::semantics) const DI: Mnemonic = Mnemonic::Nullary(Nullary::Di);
pub(in crate::analysis::semantics) const EI: Mnemonic = Mnemonic::Nullary(Nullary::Ei);
pub(in crate::analysis::semantics) const HALT: Mnemonic = Mnemonic::Nullary(Nullary::Halt);
pub(in crate::analysis::semantics) const INC: Mnemonic = Mnemonic::IncDec(IncDec::Inc);
pub(in crate::analysis::semantics) const JP: Mnemonic =
    Mnemonic::Branch(BranchKind::Explicit(ExplicitBranch::Jp));
pub(in crate::analysis::semantics) const JR: Mnemonic =
    Mnemonic::Branch(BranchKind::Explicit(ExplicitBranch::Jr));
pub(in crate::analysis::semantics) const LD: Mnemonic = Mnemonic::Ld;
pub(in crate::analysis::semantics) const LDHL: Mnemonic = Mnemonic::Ldhl;
pub(in crate::analysis::semantics) const NOP: Mnemonic = Mnemonic::Nullary(Nullary::Nop);
pub(in crate::analysis::semantics) const OR: Mnemonic = Mnemonic::Alu(AluOperation::Or);
pub(in crate::analysis::semantics) const POP: Mnemonic = Mnemonic::Stack(StackOperation::Pop);
pub(in crate::analysis::semantics) const PUSH: Mnemonic = Mnemonic::Stack(StackOperation::Push);
pub(in crate::analysis::semantics) const RES: Mnemonic = Mnemonic::Bit(BitOperation::Res);
pub(in crate::analysis::semantics) const RET: Mnemonic =
    Mnemonic::Branch(BranchKind::Implicit(ImplicitBranch::Ret));
pub(in crate::analysis::semantics) const RETI: Mnemonic =
    Mnemonic::Branch(BranchKind::Implicit(ImplicitBranch::Reti));
pub(in crate::analysis::semantics) const RL: Mnemonic = Mnemonic::Misc(MiscOperation::Rl);
pub(in crate::analysis::semantics) const RLA: Mnemonic = Mnemonic::Nullary(Nullary::Rla);
pub(in crate::analysis::semantics) const RLC: Mnemonic = Mnemonic::Misc(MiscOperation::Rlc);
pub(in crate::analysis::semantics) const RLCA: Mnemonic = Mnemonic::Nullary(Nullary::Rlca);
pub(in crate::analysis::semantics) const RR: Mnemonic = Mnemonic::Misc(MiscOperation::Rr);
pub(in crate::analysis::semantics) const RRA: Mnemonic = Mnemonic::Nullary(Nullary::Rra);
pub(in crate::analysis::semantics) const RRC: Mnemonic = Mnemonic::Misc(MiscOperation::Rrc);
pub(in crate::analysis::semantics) const RRCA: Mnemonic = Mnemonic::Nullary(Nullary::Rrca);
pub(in crate::analysis::semantics) const RST: Mnemonic = Mnemonic::Rst;
pub(in crate::analysis::semantics) const SBC: Mnemonic = Mnemonic::Alu(AluOperation::Sbc);
pub(in crate::analysis::semantics) const SET: Mnemonic = Mnemonic::Bit(BitOperation::Set);
pub(in crate::analysis::semantics) const SLA: Mnemonic = Mnemonic::Misc(MiscOperation::Sla);
pub(in crate::analysis::semantics) const SRA: Mnemonic = Mnemonic::Misc(MiscOperation::Sra);
pub(in crate::analysis::semantics) const SRL: Mnemonic = Mnemonic::Misc(MiscOperation::Srl);
pub(in crate::analysis::semantics) const STOP: Mnemonic = Mnemonic::Nullary(Nullary::Stop);
pub(in crate::analysis::semantics) const SUB: Mnemonic = Mnemonic::Alu(AluOperation::Sub);
pub(in crate::analysis::semantics) const SWAP: Mnemonic = Mnemonic::Misc(MiscOperation::Swap);
pub(in crate::analysis::semantics) const XOR: Mnemonic = Mnemonic::Alu(AluOperation::Xor);
