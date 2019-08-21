use super::instr_line::builtin_instr::cpu_instr::mnemonic::*;
use super::instr_line::builtin_instr::directive::Directive::*;
use super::instr_line::builtin_instr::operand::OperandSymbol::*;
use super::instr_line::BuiltinInstr::*;
use super::{Keyword, Keyword::*};

pub(in crate::analyze) const KEYWORDS: &[(&str, Keyword)] = &[
    ("A", Operand(A)),
    ("ADC", BuiltinInstr(Mnemonic(ADC))),
    ("ADD", BuiltinInstr(Mnemonic(ADD))),
    ("AF", Operand(Af)),
    ("AND", BuiltinInstr(Mnemonic(AND))),
    ("B", Operand(B)),
    ("BC", Operand(Bc)),
    ("BIT", BuiltinInstr(Mnemonic(BIT))),
    ("C", Operand(C)),
    ("CALL", BuiltinInstr(Mnemonic(CALL))),
    ("CP", BuiltinInstr(Mnemonic(CP))),
    ("CPL", BuiltinInstr(Mnemonic(CPL))),
    ("D", Operand(D)),
    ("DAA", BuiltinInstr(Mnemonic(DAA))),
    ("DB", BuiltinInstr(Directive(Db))),
    ("DE", Operand(De)),
    ("DEC", BuiltinInstr(Mnemonic(DEC))),
    ("DI", BuiltinInstr(Mnemonic(DI))),
    ("DS", BuiltinInstr(Directive(Ds))),
    ("DW", BuiltinInstr(Directive(Dw))),
    ("E", Operand(E)),
    ("EI", BuiltinInstr(Mnemonic(EI))),
    ("EQU", BuiltinInstr(Directive(Equ))),
    ("H", Operand(H)),
    ("HALT", BuiltinInstr(Mnemonic(HALT))),
    ("HL", Operand(Hl)),
    ("HLD", Operand(Hld)),
    ("HLI", Operand(Hli)),
    ("INC", BuiltinInstr(Mnemonic(INC))),
    ("INCLUDE", BuiltinInstr(Directive(Include))),
    ("JP", BuiltinInstr(Mnemonic(JP))),
    ("JR", BuiltinInstr(Mnemonic(JR))),
    ("L", Operand(L)),
    ("LD", BuiltinInstr(Mnemonic(LD))),
    ("LDHL", BuiltinInstr(Mnemonic(LDHL))),
    ("MACRO", BuiltinInstr(Directive(Macro))),
    ("NC", Operand(Nc)),
    ("NOP", BuiltinInstr(Mnemonic(NOP))),
    ("NZ", Operand(Nz)),
    ("OR", BuiltinInstr(Mnemonic(OR))),
    ("ORG", BuiltinInstr(Directive(Org))),
    ("POP", BuiltinInstr(Mnemonic(POP))),
    ("PUSH", BuiltinInstr(Mnemonic(PUSH))),
    ("RES", BuiltinInstr(Mnemonic(RES))),
    ("RET", BuiltinInstr(Mnemonic(RET))),
    ("RETI", BuiltinInstr(Mnemonic(RETI))),
    ("RL", BuiltinInstr(Mnemonic(RL))),
    ("RLA", BuiltinInstr(Mnemonic(RLA))),
    ("RLC", BuiltinInstr(Mnemonic(RLC))),
    ("RLCA", BuiltinInstr(Mnemonic(RLCA))),
    ("RR", BuiltinInstr(Mnemonic(RR))),
    ("RRA", BuiltinInstr(Mnemonic(RRA))),
    ("RRC", BuiltinInstr(Mnemonic(RRC))),
    ("RRCA", BuiltinInstr(Mnemonic(RRCA))),
    ("RST", BuiltinInstr(Mnemonic(RST))),
    ("SBC", BuiltinInstr(Mnemonic(SBC))),
    ("SECTION", BuiltinInstr(Directive(Section))),
    ("SET", BuiltinInstr(Mnemonic(SET))),
    ("SLA", BuiltinInstr(Mnemonic(SLA))),
    ("SP", Operand(Sp)),
    ("SRA", BuiltinInstr(Mnemonic(SRA))),
    ("SRL", BuiltinInstr(Mnemonic(SRL))),
    ("STOP", BuiltinInstr(Mnemonic(STOP))),
    ("SUB", BuiltinInstr(Mnemonic(SUB))),
    ("SWAP", BuiltinInstr(Mnemonic(SWAP))),
    ("XOR", BuiltinInstr(Mnemonic(XOR))),
    ("Z", Operand(Z)),
];
