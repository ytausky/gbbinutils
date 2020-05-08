use self::BindingDirective::*;
use self::BuiltinMnemonic::*;
use self::FreeBuiltinMnemonic::*;
use self::FreeDirective::*;

use super::arg::OperandSymbol::*;
use super::builtin_instr::cpu_instr::mnemonic::*;
use super::Keyword;
use super::Keyword::*;

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
