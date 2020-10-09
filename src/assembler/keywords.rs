use self::BuiltinMnemonic::*;
use self::Directive::*;
use self::Keyword::*;
use self::OperandKeyword::*;

use crate::IncDec;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Keyword {
    BuiltinMnemonic(BuiltinMnemonic),
    Operand(OperandKeyword),
}

pub(super) const KEYWORDS: &[(&str, Keyword)] = &[
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
pub(crate) enum BuiltinMnemonic {
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
            BuiltinMnemonic::Directive(directive) => matches!(
                directive,
                Directive::Equ | Directive::Macro | Directive::Section
            ),
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Mnemonic {
    Alu(AluOperation),
    Bit(BitOperation),
    Branch(BranchKind),
    IncDec(IncDec),
    Ld,
    Ldhl,
    Misc(MiscOperation),
    Nullary(u8),
    Rst,
    Stack(StackOperation),
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AluOperation {
    Add,
    Adc,
    Sub,
    Sbc,
    And,
    Xor,
    Or,
    Cp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BitOperation {
    Bit,
    Set,
    Res,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MiscOperation {
    Rlc,
    Rrc,
    Rl,
    Rr,
    Sla,
    Sra,
    Swap,
    Srl,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StackOperation {
    Push,
    Pop,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OperandKeyword {
    A,
    Af,
    B,
    Bc,
    C,
    D,
    De,
    E,
    H,
    Hl,
    Hld,
    Hli,
    L,
    Nc,
    Nz,
    Sp,
    Z,
}

pub(super) const ADC: Mnemonic = Mnemonic::Alu(AluOperation::Adc);
pub(super) const ADD: Mnemonic = Mnemonic::Alu(AluOperation::Add);
pub(super) const AND: Mnemonic = Mnemonic::Alu(AluOperation::And);
pub(super) const BIT: Mnemonic = Mnemonic::Bit(BitOperation::Bit);
pub(super) const CALL: Mnemonic = Mnemonic::Branch(BranchKind::Explicit(ExplicitBranch::Call));
pub(super) const CP: Mnemonic = Mnemonic::Alu(AluOperation::Cp);
pub(super) const CPL: Mnemonic = Mnemonic::Nullary(0x2f);
pub(super) const DAA: Mnemonic = Mnemonic::Nullary(0x27);
pub(super) const DEC: Mnemonic = Mnemonic::IncDec(IncDec::Dec);
pub(super) const DI: Mnemonic = Mnemonic::Nullary(0xf3);
pub(super) const EI: Mnemonic = Mnemonic::Nullary(0xfb);
pub(super) const HALT: Mnemonic = Mnemonic::Nullary(0x76);
pub(super) const INC: Mnemonic = Mnemonic::IncDec(IncDec::Inc);
pub(super) const JP: Mnemonic = Mnemonic::Branch(BranchKind::Explicit(ExplicitBranch::Jp));
pub(super) const JR: Mnemonic = Mnemonic::Branch(BranchKind::Explicit(ExplicitBranch::Jr));
pub(super) const LD: Mnemonic = Mnemonic::Ld;
pub(super) const LDHL: Mnemonic = Mnemonic::Ldhl;
pub(super) const NOP: Mnemonic = Mnemonic::Nullary(0x00);
pub(super) const OR: Mnemonic = Mnemonic::Alu(AluOperation::Or);
pub(super) const POP: Mnemonic = Mnemonic::Stack(StackOperation::Pop);
pub(super) const PUSH: Mnemonic = Mnemonic::Stack(StackOperation::Push);
pub(super) const RES: Mnemonic = Mnemonic::Bit(BitOperation::Res);
pub(super) const RET: Mnemonic = Mnemonic::Branch(BranchKind::Implicit(ImplicitBranch::Ret));
pub(super) const RETI: Mnemonic = Mnemonic::Branch(BranchKind::Implicit(ImplicitBranch::Reti));
pub(super) const RL: Mnemonic = Mnemonic::Misc(MiscOperation::Rl);
pub(super) const RLA: Mnemonic = Mnemonic::Nullary(0x17);
pub(super) const RLC: Mnemonic = Mnemonic::Misc(MiscOperation::Rlc);
pub(super) const RLCA: Mnemonic = Mnemonic::Nullary(0x07);
pub(super) const RR: Mnemonic = Mnemonic::Misc(MiscOperation::Rr);
pub(super) const RRA: Mnemonic = Mnemonic::Nullary(0x1f);
pub(super) const RRC: Mnemonic = Mnemonic::Misc(MiscOperation::Rrc);
pub(super) const RRCA: Mnemonic = Mnemonic::Nullary(0x0f);
pub(super) const RST: Mnemonic = Mnemonic::Rst;
pub(super) const SBC: Mnemonic = Mnemonic::Alu(AluOperation::Sbc);
pub(super) const SET: Mnemonic = Mnemonic::Bit(BitOperation::Set);
pub(super) const SLA: Mnemonic = Mnemonic::Misc(MiscOperation::Sla);
pub(super) const SRA: Mnemonic = Mnemonic::Misc(MiscOperation::Sra);
pub(super) const SRL: Mnemonic = Mnemonic::Misc(MiscOperation::Srl);
pub(super) const STOP: Mnemonic = Mnemonic::Stop;
pub(super) const SUB: Mnemonic = Mnemonic::Alu(AluOperation::Sub);
pub(super) const SWAP: Mnemonic = Mnemonic::Misc(MiscOperation::Swap);
pub(super) const XOR: Mnemonic = Mnemonic::Alu(AluOperation::Xor);

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
