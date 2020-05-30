use self::BuiltinMnemonic::*;
use self::Directive::*;

use super::arg::OperandSymbol::*;
use super::Keyword;
use super::Keyword::*;

use crate::object::builder::*;

pub const KEYWORDS: &[(&str, Keyword)] = &[
    ("A", Operand(A)),
    ("ADC", BuiltinMnemonic(CpuInstr(ADC))),
    ("ADD", BuiltinMnemonic(CpuInstr(ADD))),
    ("AF", Operand(Af)),
    ("AND", BuiltinMnemonic(CpuInstr(AND))),
    ("B", Operand(B)),
    ("BC", Operand(Bc)),
    ("BIT", BuiltinMnemonic(CpuInstr(BIT))),
    ("C", Operand(C)),
    ("CALL", BuiltinMnemonic(CpuInstr(CALL))),
    ("CP", BuiltinMnemonic(CpuInstr(CP))),
    ("CPL", BuiltinMnemonic(CpuInstr(CPL))),
    ("D", Operand(D)),
    ("DAA", BuiltinMnemonic(CpuInstr(DAA))),
    ("DB", BuiltinMnemonic(Directive(Db))),
    ("DE", Operand(De)),
    ("DEC", BuiltinMnemonic(CpuInstr(DEC))),
    ("DI", BuiltinMnemonic(CpuInstr(DI))),
    ("DS", BuiltinMnemonic(Directive(Ds))),
    ("DW", BuiltinMnemonic(Directive(Dw))),
    ("E", Operand(E)),
    ("EI", BuiltinMnemonic(CpuInstr(EI))),
    ("ENDC", BuiltinMnemonic(Directive(Endc))),
    ("ENDM", BuiltinMnemonic(Directive(Endm))),
    ("EQU", BuiltinMnemonic(Directive(Equ))),
    ("H", Operand(H)),
    ("HALT", BuiltinMnemonic(CpuInstr(HALT))),
    ("HL", Operand(Hl)),
    ("HLD", Operand(Hld)),
    ("HLI", Operand(Hli)),
    ("IF", BuiltinMnemonic(Directive(If))),
    ("INC", BuiltinMnemonic(CpuInstr(INC))),
    ("INCLUDE", BuiltinMnemonic(Directive(Include))),
    ("JP", BuiltinMnemonic(CpuInstr(JP))),
    ("JR", BuiltinMnemonic(CpuInstr(JR))),
    ("L", Operand(L)),
    ("LD", BuiltinMnemonic(CpuInstr(LD))),
    ("LDHL", BuiltinMnemonic(CpuInstr(LDHL))),
    ("MACRO", BuiltinMnemonic(Directive(Macro))),
    ("NC", Operand(Nc)),
    ("NOP", BuiltinMnemonic(CpuInstr(NOP))),
    ("NZ", Operand(Nz)),
    ("OR", BuiltinMnemonic(CpuInstr(OR))),
    ("ORG", BuiltinMnemonic(Directive(Org))),
    ("POP", BuiltinMnemonic(CpuInstr(POP))),
    ("PUSH", BuiltinMnemonic(CpuInstr(PUSH))),
    ("RES", BuiltinMnemonic(CpuInstr(RES))),
    ("RET", BuiltinMnemonic(CpuInstr(RET))),
    ("RETI", BuiltinMnemonic(CpuInstr(RETI))),
    ("RL", BuiltinMnemonic(CpuInstr(RL))),
    ("RLA", BuiltinMnemonic(CpuInstr(RLA))),
    ("RLC", BuiltinMnemonic(CpuInstr(RLC))),
    ("RLCA", BuiltinMnemonic(CpuInstr(RLCA))),
    ("RR", BuiltinMnemonic(CpuInstr(RR))),
    ("RRA", BuiltinMnemonic(CpuInstr(RRA))),
    ("RRC", BuiltinMnemonic(CpuInstr(RRC))),
    ("RRCA", BuiltinMnemonic(CpuInstr(RRCA))),
    ("RST", BuiltinMnemonic(CpuInstr(RST))),
    ("SBC", BuiltinMnemonic(CpuInstr(SBC))),
    ("SECTION", BuiltinMnemonic(Directive(Section))),
    ("SET", BuiltinMnemonic(CpuInstr(SET))),
    ("SLA", BuiltinMnemonic(CpuInstr(SLA))),
    ("SP", Operand(Sp)),
    ("SRA", BuiltinMnemonic(CpuInstr(SRA))),
    ("SRL", BuiltinMnemonic(CpuInstr(SRL))),
    ("STOP", BuiltinMnemonic(CpuInstr(STOP))),
    ("SUB", BuiltinMnemonic(CpuInstr(SUB))),
    ("SWAP", BuiltinMnemonic(CpuInstr(SWAP))),
    ("XOR", BuiltinMnemonic(CpuInstr(XOR))),
    ("Z", Operand(Z)),
];

#[derive(Clone, Debug, PartialEq)]
pub enum BuiltinMnemonic {
    CpuInstr(Mnemonic),
    Directive(Directive),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Directive {
    Db,
    Ds,
    Dw,
    Endc,
    Endm,
    Equ,
    If,
    Include,
    Macro,
    Org,
    Section,
}

impl BuiltinMnemonic {
    pub fn binds_to_label(&self) -> bool {
        match self {
            BuiltinMnemonic::Directive(directive) => match directive {
                Directive::Equ | Directive::Macro | Directive::Section => true,
                _ => false,
            },
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Mnemonic {
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
pub enum StackOperation {
    Push,
    Pop,
}

pub(crate) const ADC: Mnemonic = Mnemonic::Alu(AluOperation::Adc);
pub(crate) const ADD: Mnemonic = Mnemonic::Alu(AluOperation::Add);
pub(crate) const AND: Mnemonic = Mnemonic::Alu(AluOperation::And);
pub(crate) const BIT: Mnemonic = Mnemonic::Bit(BitOperation::Bit);
pub(crate) const CALL: Mnemonic = Mnemonic::Branch(BranchKind::Explicit(ExplicitBranch::Call));
pub(crate) const CP: Mnemonic = Mnemonic::Alu(AluOperation::Cp);
pub(crate) const CPL: Mnemonic = Mnemonic::Nullary(Nullary::Cpl);
pub(crate) const DAA: Mnemonic = Mnemonic::Nullary(Nullary::Daa);
pub(crate) const DEC: Mnemonic = Mnemonic::IncDec(IncDec::Dec);
pub(crate) const DI: Mnemonic = Mnemonic::Nullary(Nullary::Di);
pub(crate) const EI: Mnemonic = Mnemonic::Nullary(Nullary::Ei);
pub(crate) const HALT: Mnemonic = Mnemonic::Nullary(Nullary::Halt);
pub(crate) const INC: Mnemonic = Mnemonic::IncDec(IncDec::Inc);
pub(crate) const JP: Mnemonic = Mnemonic::Branch(BranchKind::Explicit(ExplicitBranch::Jp));
pub(crate) const JR: Mnemonic = Mnemonic::Branch(BranchKind::Explicit(ExplicitBranch::Jr));
pub(crate) const LD: Mnemonic = Mnemonic::Ld;
pub(crate) const LDHL: Mnemonic = Mnemonic::Ldhl;
pub(crate) const NOP: Mnemonic = Mnemonic::Nullary(Nullary::Nop);
pub(crate) const OR: Mnemonic = Mnemonic::Alu(AluOperation::Or);
pub(crate) const POP: Mnemonic = Mnemonic::Stack(StackOperation::Pop);
pub(crate) const PUSH: Mnemonic = Mnemonic::Stack(StackOperation::Push);
pub(crate) const RES: Mnemonic = Mnemonic::Bit(BitOperation::Res);
pub(crate) const RET: Mnemonic = Mnemonic::Branch(BranchKind::Implicit(ImplicitBranch::Ret));
pub(crate) const RETI: Mnemonic = Mnemonic::Branch(BranchKind::Implicit(ImplicitBranch::Reti));
pub(crate) const RL: Mnemonic = Mnemonic::Misc(MiscOperation::Rl);
pub(crate) const RLA: Mnemonic = Mnemonic::Nullary(Nullary::Rla);
pub(crate) const RLC: Mnemonic = Mnemonic::Misc(MiscOperation::Rlc);
pub(crate) const RLCA: Mnemonic = Mnemonic::Nullary(Nullary::Rlca);
pub(crate) const RR: Mnemonic = Mnemonic::Misc(MiscOperation::Rr);
pub(crate) const RRA: Mnemonic = Mnemonic::Nullary(Nullary::Rra);
pub(crate) const RRC: Mnemonic = Mnemonic::Misc(MiscOperation::Rrc);
pub(crate) const RRCA: Mnemonic = Mnemonic::Nullary(Nullary::Rrca);
pub(crate) const RST: Mnemonic = Mnemonic::Rst;
pub(crate) const SBC: Mnemonic = Mnemonic::Alu(AluOperation::Sbc);
pub(crate) const SET: Mnemonic = Mnemonic::Bit(BitOperation::Set);
pub(crate) const SLA: Mnemonic = Mnemonic::Misc(MiscOperation::Sla);
pub(crate) const SRA: Mnemonic = Mnemonic::Misc(MiscOperation::Sra);
pub(crate) const SRL: Mnemonic = Mnemonic::Misc(MiscOperation::Srl);
pub(crate) const STOP: Mnemonic = Mnemonic::Nullary(Nullary::Stop);
pub(crate) const SUB: Mnemonic = Mnemonic::Alu(AluOperation::Sub);
pub(crate) const SWAP: Mnemonic = Mnemonic::Misc(MiscOperation::Swap);
pub(crate) const XOR: Mnemonic = Mnemonic::Alu(AluOperation::Xor);

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
