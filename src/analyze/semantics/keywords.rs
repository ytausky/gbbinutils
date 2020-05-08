use self::BindingDirective::*;
use self::BuiltinMnemonic::*;
use self::FreeBuiltinMnemonic::*;
use self::FreeDirective::*;

use super::arg::OperandSymbol::*;
use super::Keyword;
use super::Keyword::*;

use crate::object::builder::*;

pub(super) const KEYWORDS: &[(&str, Keyword)] = &[
    ("A", Operand(A)),
    ("ADC", BuiltinMnemonic(Free(CpuInstr(ADC)))),
    ("ADD", BuiltinMnemonic(Free(CpuInstr(ADD)))),
    ("AF", Operand(Af)),
    ("AND", BuiltinMnemonic(Free(CpuInstr(AND)))),
    ("B", Operand(B)),
    ("BC", Operand(Bc)),
    ("BIT", BuiltinMnemonic(Free(CpuInstr(BIT)))),
    ("C", Operand(C)),
    ("CALL", BuiltinMnemonic(Free(CpuInstr(CALL)))),
    ("CP", BuiltinMnemonic(Free(CpuInstr(CP)))),
    ("CPL", BuiltinMnemonic(Free(CpuInstr(CPL)))),
    ("D", Operand(D)),
    ("DAA", BuiltinMnemonic(Free(CpuInstr(DAA)))),
    ("DB", BuiltinMnemonic(Free(Directive(Db)))),
    ("DE", Operand(De)),
    ("DEC", BuiltinMnemonic(Free(CpuInstr(DEC)))),
    ("DI", BuiltinMnemonic(Free(CpuInstr(DI)))),
    ("DS", BuiltinMnemonic(Free(Directive(Ds)))),
    ("DW", BuiltinMnemonic(Free(Directive(Dw)))),
    ("E", Operand(E)),
    ("EI", BuiltinMnemonic(Free(CpuInstr(EI)))),
    ("ENDC", BuiltinMnemonic(Free(Directive(Endc)))),
    ("ENDM", BuiltinMnemonic(Free(Directive(Endm)))),
    ("EQU", BuiltinMnemonic(Binding(Equ))),
    ("H", Operand(H)),
    ("HALT", BuiltinMnemonic(Free(CpuInstr(HALT)))),
    ("HL", Operand(Hl)),
    ("HLD", Operand(Hld)),
    ("HLI", Operand(Hli)),
    ("IF", BuiltinMnemonic(Free(Directive(If)))),
    ("INC", BuiltinMnemonic(Free(CpuInstr(INC)))),
    ("INCLUDE", BuiltinMnemonic(Free(Directive(Include)))),
    ("JP", BuiltinMnemonic(Free(CpuInstr(JP)))),
    ("JR", BuiltinMnemonic(Free(CpuInstr(JR)))),
    ("L", Operand(L)),
    ("LD", BuiltinMnemonic(Free(CpuInstr(LD)))),
    ("LDHL", BuiltinMnemonic(Free(CpuInstr(LDHL)))),
    ("MACRO", BuiltinMnemonic(Binding(Macro))),
    ("NC", Operand(Nc)),
    ("NOP", BuiltinMnemonic(Free(CpuInstr(NOP)))),
    ("NZ", Operand(Nz)),
    ("OR", BuiltinMnemonic(Free(CpuInstr(OR)))),
    ("ORG", BuiltinMnemonic(Free(Directive(Org)))),
    ("POP", BuiltinMnemonic(Free(CpuInstr(POP)))),
    ("PUSH", BuiltinMnemonic(Free(CpuInstr(PUSH)))),
    ("RES", BuiltinMnemonic(Free(CpuInstr(RES)))),
    ("RET", BuiltinMnemonic(Free(CpuInstr(RET)))),
    ("RETI", BuiltinMnemonic(Free(CpuInstr(RETI)))),
    ("RL", BuiltinMnemonic(Free(CpuInstr(RL)))),
    ("RLA", BuiltinMnemonic(Free(CpuInstr(RLA)))),
    ("RLC", BuiltinMnemonic(Free(CpuInstr(RLC)))),
    ("RLCA", BuiltinMnemonic(Free(CpuInstr(RLCA)))),
    ("RR", BuiltinMnemonic(Free(CpuInstr(RR)))),
    ("RRA", BuiltinMnemonic(Free(CpuInstr(RRA)))),
    ("RRC", BuiltinMnemonic(Free(CpuInstr(RRC)))),
    ("RRCA", BuiltinMnemonic(Free(CpuInstr(RRCA)))),
    ("RST", BuiltinMnemonic(Free(CpuInstr(RST)))),
    ("SBC", BuiltinMnemonic(Free(CpuInstr(SBC)))),
    ("SECTION", BuiltinMnemonic(Binding(Section))),
    ("SET", BuiltinMnemonic(Free(CpuInstr(SET)))),
    ("SLA", BuiltinMnemonic(Free(CpuInstr(SLA)))),
    ("SP", Operand(Sp)),
    ("SRA", BuiltinMnemonic(Free(CpuInstr(SRA)))),
    ("SRL", BuiltinMnemonic(Free(CpuInstr(SRL)))),
    ("STOP", BuiltinMnemonic(Free(CpuInstr(STOP)))),
    ("SUB", BuiltinMnemonic(Free(CpuInstr(SUB)))),
    ("SWAP", BuiltinMnemonic(Free(CpuInstr(SWAP)))),
    ("XOR", BuiltinMnemonic(Free(CpuInstr(XOR)))),
    ("Z", Operand(Z)),
];

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum BuiltinMnemonic {
    Binding(BindingDirective),
    Free(FreeBuiltinMnemonic),
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum FreeBuiltinMnemonic {
    CpuInstr(Mnemonic),
    Directive(FreeDirective),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::analyze) enum BindingDirective {
    Equ,
    Macro,
    Section,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::analyze) enum FreeDirective {
    Db,
    Ds,
    Dw,
    Endc,
    Endm,
    If,
    Include,
    Org,
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum Mnemonic {
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
pub(in crate::analyze) enum StackOperation {
    Push,
    Pop,
}

pub(in crate::analyze::semantics) const ADC: Mnemonic = Mnemonic::Alu(AluOperation::Adc);
pub(in crate::analyze::semantics) const ADD: Mnemonic = Mnemonic::Alu(AluOperation::Add);
pub(in crate::analyze::semantics) const AND: Mnemonic = Mnemonic::Alu(AluOperation::And);
pub(in crate::analyze::semantics) const BIT: Mnemonic = Mnemonic::Bit(BitOperation::Bit);
pub(in crate::analyze::semantics) const CALL: Mnemonic =
    Mnemonic::Branch(BranchKind::Explicit(ExplicitBranch::Call));
pub(in crate::analyze::semantics) const CP: Mnemonic = Mnemonic::Alu(AluOperation::Cp);
pub(in crate::analyze::semantics) const CPL: Mnemonic = Mnemonic::Nullary(Nullary::Cpl);
pub(in crate::analyze::semantics) const DAA: Mnemonic = Mnemonic::Nullary(Nullary::Daa);
pub(in crate::analyze::semantics) const DEC: Mnemonic = Mnemonic::IncDec(IncDec::Dec);
pub(in crate::analyze::semantics) const DI: Mnemonic = Mnemonic::Nullary(Nullary::Di);
pub(in crate::analyze::semantics) const EI: Mnemonic = Mnemonic::Nullary(Nullary::Ei);
pub(in crate::analyze::semantics) const HALT: Mnemonic = Mnemonic::Nullary(Nullary::Halt);
pub(in crate::analyze::semantics) const INC: Mnemonic = Mnemonic::IncDec(IncDec::Inc);
pub(in crate::analyze::semantics) const JP: Mnemonic =
    Mnemonic::Branch(BranchKind::Explicit(ExplicitBranch::Jp));
pub(in crate::analyze::semantics) const JR: Mnemonic =
    Mnemonic::Branch(BranchKind::Explicit(ExplicitBranch::Jr));
pub(in crate::analyze::semantics) const LD: Mnemonic = Mnemonic::Ld;
pub(in crate::analyze::semantics) const LDHL: Mnemonic = Mnemonic::Ldhl;
pub(in crate::analyze::semantics) const NOP: Mnemonic = Mnemonic::Nullary(Nullary::Nop);
pub(in crate::analyze::semantics) const OR: Mnemonic = Mnemonic::Alu(AluOperation::Or);
pub(in crate::analyze::semantics) const POP: Mnemonic = Mnemonic::Stack(StackOperation::Pop);
pub(in crate::analyze::semantics) const PUSH: Mnemonic = Mnemonic::Stack(StackOperation::Push);
pub(in crate::analyze::semantics) const RES: Mnemonic = Mnemonic::Bit(BitOperation::Res);
pub(in crate::analyze::semantics) const RET: Mnemonic =
    Mnemonic::Branch(BranchKind::Implicit(ImplicitBranch::Ret));
pub(in crate::analyze::semantics) const RETI: Mnemonic =
    Mnemonic::Branch(BranchKind::Implicit(ImplicitBranch::Reti));
pub(in crate::analyze::semantics) const RL: Mnemonic = Mnemonic::Misc(MiscOperation::Rl);
pub(in crate::analyze::semantics) const RLA: Mnemonic = Mnemonic::Nullary(Nullary::Rla);
pub(in crate::analyze::semantics) const RLC: Mnemonic = Mnemonic::Misc(MiscOperation::Rlc);
pub(in crate::analyze::semantics) const RLCA: Mnemonic = Mnemonic::Nullary(Nullary::Rlca);
pub(in crate::analyze::semantics) const RR: Mnemonic = Mnemonic::Misc(MiscOperation::Rr);
pub(in crate::analyze::semantics) const RRA: Mnemonic = Mnemonic::Nullary(Nullary::Rra);
pub(in crate::analyze::semantics) const RRC: Mnemonic = Mnemonic::Misc(MiscOperation::Rrc);
pub(in crate::analyze::semantics) const RRCA: Mnemonic = Mnemonic::Nullary(Nullary::Rrca);
pub(in crate::analyze::semantics) const RST: Mnemonic = Mnemonic::Rst;
pub(in crate::analyze::semantics) const SBC: Mnemonic = Mnemonic::Alu(AluOperation::Sbc);
pub(in crate::analyze::semantics) const SET: Mnemonic = Mnemonic::Bit(BitOperation::Set);
pub(in crate::analyze::semantics) const SLA: Mnemonic = Mnemonic::Misc(MiscOperation::Sla);
pub(in crate::analyze::semantics) const SRA: Mnemonic = Mnemonic::Misc(MiscOperation::Sra);
pub(in crate::analyze::semantics) const SRL: Mnemonic = Mnemonic::Misc(MiscOperation::Srl);
pub(in crate::analyze::semantics) const STOP: Mnemonic = Mnemonic::Nullary(Nullary::Stop);
pub(in crate::analyze::semantics) const SUB: Mnemonic = Mnemonic::Alu(AluOperation::Sub);
pub(in crate::analyze::semantics) const SWAP: Mnemonic = Mnemonic::Misc(MiscOperation::Swap);
pub(in crate::analyze::semantics) const XOR: Mnemonic = Mnemonic::Alu(AluOperation::Xor);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BranchKind {
    Explicit(ExplicitBranch),
    Implicit(ImplicitBranch),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExplicitBranch {
    Call,
    Jp,
    Jr,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImplicitBranch {
    Ret,
    Reti,
}
