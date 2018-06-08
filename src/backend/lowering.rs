use backend::Item;
use instruction::*;
use std::mem;
use Width;

#[derive(Debug, PartialEq)]
pub enum DataItem<R> {
    Byte(u8),
    Expr(RelocExpr<R>, Width),
}

pub trait Lower<SR> {
    fn lower(self) -> LoweredItem<SR>;
}

pub enum LoweredItem<SR> {
    None,
    One(DataItem<SR>),
    Two(DataItem<SR>, DataItem<SR>),
}

impl<SR> Iterator for LoweredItem<SR> {
    type Item = DataItem<SR>;
    fn next(&mut self) -> Option<Self::Item> {
        match mem::replace(self, LoweredItem::None) {
            LoweredItem::None => None,
            LoweredItem::One(item) => Some(item),
            LoweredItem::Two(first, second) => {
                *self = LoweredItem::One(second);
                Some(first)
            }
        }
    }
}

impl<SR> LoweredItem<SR> {
    fn with_opcode(opcode: u8) -> LoweredItem<SR> {
        LoweredItem::One(DataItem::Byte(opcode))
    }

    fn and_byte(self, expr: RelocExpr<SR>) -> Self {
        self.and_expr(expr, Width::Byte)
    }

    fn and_word(self, expr: RelocExpr<SR>) -> Self {
        self.and_expr(expr, Width::Word)
    }

    fn and_expr(self, expr: RelocExpr<SR>, width: Width) -> Self {
        match self {
            LoweredItem::One(item) => LoweredItem::Two(item, DataItem::Expr(expr, width)),
            LoweredItem::None | LoweredItem::Two(..) => panic!(),
        }
    }
}

impl<SR> Lower<SR> for Item<SR> {
    fn lower(self) -> LoweredItem<SR> {
        match self {
            Item::Data(expr, width) => LoweredItem::One(DataItem::Expr(expr, width)),
            Item::Instruction(instruction) => instruction.lower(),
        }
    }
}

impl<SR> Lower<SR> for Instruction<SR> {
    fn lower(self) -> LoweredItem<SR> {
        use instruction::Instruction::*;
        match self {
            AddHl(reg16) => LoweredItem::with_opcode(0x09 | encode_reg16(reg16)),
            Alu(operation, AluSource::Simple(src)) => encode_simple_alu_operation(operation, src),
            Alu(operation, AluSource::Immediate(expr)) => {
                encode_immediate_alu_operation(operation, expr)
            }
            Branch(branch, condition) => encode_branch(branch, condition),
            IncDec8(mode, operand) => LoweredItem::with_opcode(
                0b00_000_100 | encode_inc_dec(mode) | (encode_simple_operand(operand) << 3),
            ),
            IncDec16(mode, operand) => {
                LoweredItem::with_opcode(0x03 | (encode_inc_dec(mode) << 3) | encode_reg16(operand))
            }
            JpDerefHl => LoweredItem::with_opcode(0xe9),
            Ld(ld) => ld.lower(),
            Nullary(nullary) => nullary.lower(),
            Pop(reg_pair) => LoweredItem::with_opcode(0xc1 | (encode_reg_pair(reg_pair) << 4)),
            Push(reg_pair) => LoweredItem::with_opcode(0xc5 | (encode_reg_pair(reg_pair) << 4)),
        }
    }
}

impl<SR> Lower<SR> for Nullary {
    fn lower(self) -> LoweredItem<SR> {
        use instruction::Nullary::*;
        let opcode = match self {
            Daa => 0x27,
            Di => 0xf3,
            Ei => 0xfb,
            Halt => 0x76,
            Nop => 0x00,
            Stop => 0x10,
            Reti => 0xd9,
        };
        if self == Stop {
            LoweredItem::Two(DataItem::Byte(opcode), DataItem::Byte(0x00))
        } else {
            LoweredItem::with_opcode(opcode)
        }
    }
}

impl<SR> Lower<SR> for Ld<SR> {
    fn lower(self) -> LoweredItem<SR> {
        match self {
            Ld::Simple(dest, src) => encode_ld_to_reg_from_reg(dest, src),
            Ld::Special(special, direction) => encode_special_ld(special, direction),
            Ld::Immediate8(dest, immediate) => LoweredItem::with_opcode(
                0x06 | (encode_simple_operand(dest) << 3),
            ).and_byte(immediate),
            Ld::Immediate16(dest, immediate) => {
                LoweredItem::with_opcode(0x01 | encode_reg16(dest)).and_word(immediate)
            }
        }
    }
}

fn encode_special_ld<SR>(ld: SpecialLd<SR>, direction: Direction) -> LoweredItem<SR> {
    match ld {
        SpecialLd::InlineAddr(addr) => {
            LoweredItem::with_opcode(0xea | encode_direction(direction)).and_word(addr)
        }
        SpecialLd::InlineIndex(index) => {
            LoweredItem::with_opcode(0xe0 | encode_direction(direction)).and_byte(index)
        }
        SpecialLd::RegIndex => LoweredItem::with_opcode(0xe2 | encode_direction(direction)),
    }
}

fn encode_simple_alu_operation<SR>(operation: AluOperation, src: SimpleOperand) -> LoweredItem<SR> {
    use instruction::AluOperation::*;
    let opcode_base = match operation {
        Add => 0x80,
        And => 0xa0,
        Cp => 0xb8,
        Xor => 0xa8,
    };
    LoweredItem::with_opcode(opcode_base | encode_simple_operand(src))
}

fn encode_immediate_alu_operation<SR>(
    operation: AluOperation,
    expr: RelocExpr<SR>,
) -> LoweredItem<SR> {
    use instruction::AluOperation::*;
    LoweredItem::with_opcode(match operation {
        Add => 0xc6,
        And => 0xe6,
        Cp => 0xfe,
        Xor => 0xee,
    }).and_byte(expr)
}

fn encode_branch<SR>(branch: Branch<SR>, condition: Option<Condition>) -> LoweredItem<SR> {
    use instruction::Branch::*;
    match branch {
        Call(target) => LoweredItem::with_opcode(match condition {
            None => 0xcd,
            Some(condition) => 0xc4 | encode_condition(condition),
        }).and_word(target),
        Jp(target) => LoweredItem::with_opcode(match condition {
            None => 0xc3,
            Some(condition) => 0xc2 | encode_condition(condition),
        }).and_word(target),
        Jr(target) => LoweredItem::with_opcode(match condition {
            None => 0x18,
            Some(condition) => 0x20 | encode_condition(condition),
        }).and_byte(RelocExpr::Subtract(
            Box::new(target),
            Box::new(RelocExpr::LocationCounter),
        )),
        Ret => LoweredItem::with_opcode(match condition {
            None => 0xc9,
            Some(condition) => 0b11_000_000 | encode_condition(condition),
        }),
    }
}

fn encode_condition(condition: Condition) -> u8 {
    use instruction::Condition::*;
    (match condition {
        Nz => 0b00,
        Z => 0b01,
        Nc => 0b10,
        C => 0b11,
    }) << 3
}

fn encode_direction(direction: Direction) -> u8 {
    match direction {
        Direction::FromA => 0x00,
        Direction::IntoA => 0x10,
    }
}

fn encode_ld_to_reg_from_reg<SR>(dest: SimpleOperand, src: SimpleOperand) -> LoweredItem<SR> {
    LoweredItem::with_opcode(
        0b01_000_000 | (encode_simple_operand(dest) << 3) | encode_simple_operand(src),
    )
}

fn encode_simple_operand(register: SimpleOperand) -> u8 {
    use instruction::SimpleOperand::*;
    match register {
        A => 0b111,
        B => 0b000,
        C => 0b001,
        D => 0b010,
        E => 0b011,
        H => 0b100,
        L => 0b101,
        DerefHl => 0b110,
    }
}

fn encode_reg16(reg16: Reg16) -> u8 {
    use instruction::Reg16::*;
    (match reg16 {
        Bc => 0b00,
        De => 0b01,
        Hl => 0b10,
        Sp => 0b11,
    }) << 4
}

fn encode_reg_pair(reg_pair: RegPair) -> u8 {
    use instruction::RegPair::*;
    match reg_pair {
        Bc => 0b00,
        De => 0b01,
        Hl => 0b10,
        Af => 0b11,
    }
}

fn encode_inc_dec(mode: IncDec) -> u8 {
    use instruction::IncDec::*;
    match mode {
        Inc => 0,
        Dec => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Borrow;

    use instruction::{self, Branch::*, Instruction::*, Ld::*, Nullary::*};

    fn test_instruction(instruction: Instruction<()>, data_items: impl Borrow<[DataItem<()>]>) {
        let mut code = vec![];
        code.extend(instruction.lower());
        assert_eq!(code, data_items.borrow())
    }

    #[test]
    fn encode_nop() {
        test_nullary(Nop, bytes([0x00]))
    }

    #[test]
    fn encode_stop() {
        test_nullary(Stop, bytes([0x10, 0x00]))
    }

    #[test]
    fn encode_halt() {
        test_nullary(Halt, bytes([0x76]))
    }

    #[test]
    fn encode_di() {
        test_nullary(Di, bytes([0xf3]))
    }

    #[test]
    fn encode_ei() {
        test_nullary(Ei, bytes([0xfb]))
    }

    #[test]
    fn encode_daa() {
        test_nullary(Daa, bytes([0x27]))
    }

    fn test_nullary(nullary: instruction::Nullary, items: impl Borrow<[DataItem<()>]>) {
        test_instruction(Nullary(nullary), items)
    }

    #[test]
    fn encode_8_bit_register_transfers() {
        use self::SimpleOperand::*;
        let operands_and_encoding = vec![
            (A, A, 0x7f),
            (A, B, 0x78),
            (A, C, 0x79),
            (A, D, 0x7a),
            (A, E, 0x7b),
            (A, H, 0x7c),
            (A, L, 0x7d),
            (B, A, 0x47),
            (B, B, 0x40),
            (B, C, 0x41),
            (B, D, 0x42),
            (B, E, 0x43),
            (B, H, 0x44),
            (B, L, 0x45),
            (C, A, 0x4f),
            (C, B, 0x48),
            (C, C, 0x49),
            (C, D, 0x4a),
            (C, E, 0x4b),
            (C, H, 0x4c),
            (C, L, 0x4d),
            (D, A, 0x57),
            (D, B, 0x50),
            (D, C, 0x51),
            (D, D, 0x52),
            (D, E, 0x53),
            (D, H, 0x54),
            (D, L, 0x55),
            (E, A, 0x5f),
            (E, B, 0x58),
            (E, C, 0x59),
            (E, D, 0x5a),
            (E, E, 0x5b),
            (E, H, 0x5c),
            (E, L, 0x5d),
            (H, A, 0x67),
            (H, B, 0x60),
            (H, C, 0x61),
            (H, D, 0x62),
            (H, E, 0x63),
            (H, H, 0x64),
            (H, L, 0x65),
            (L, A, 0x6f),
            (L, B, 0x68),
            (L, C, 0x69),
            (L, D, 0x6a),
            (L, E, 0x6b),
            (L, H, 0x6c),
            (L, L, 0x6d),
        ];
        for (dest, src, opcode) in operands_and_encoding {
            test_instruction(Ld(Simple(dest, src)), bytes([opcode]))
        }
    }

    #[test]
    fn encode_ld_simple_immediate() {
        use instruction::SimpleOperand::*;
        let immediate = RelocExpr::Literal(0x42, ());
        vec![
            (B, 0x06),
            (C, 0x0e),
            (D, 0x16),
            (E, 0x1e),
            (H, 0x26),
            (L, 0x2e),
            (DerefHl, 0x36),
            (A, 0x3e),
        ].into_iter()
            .for_each(|(dest, opcode)| {
                test_instruction(
                    Ld(Immediate8(dest, immediate.clone())),
                    [
                        DataItem::Byte(opcode),
                        DataItem::Expr(immediate.clone(), Width::Byte),
                    ],
                )
            })
    }

    #[test]
    fn encode_ld_immediate16() {
        use instruction::Reg16::*;
        let immediate = RelocExpr::Literal(0x1234, ());
        let test_cases = &[(Bc, 0x01), (De, 0x11), (Hl, 0x21), (Sp, 0x31)];
        for &(reg16, opcode) in test_cases {
            test_instruction(
                Ld(Immediate16(reg16, immediate.clone())),
                [
                    DataItem::Byte(opcode),
                    DataItem::Expr(immediate.clone(), Width::Word),
                ],
            )
        }
    }

    #[test]
    fn encode_ld_inline_addr() {
        let addr = RelocExpr::Literal(0x1234, ());
        let test_cases = &[(Direction::FromA, 0xea), (Direction::IntoA, 0xfa)];
        for &(direction, opcode) in test_cases {
            test_instruction(
                Ld(Special(SpecialLd::InlineAddr(addr.clone()), direction)),
                [
                    DataItem::Byte(opcode),
                    DataItem::Expr(addr.clone(), Width::Word),
                ],
            )
        }
    }

    #[test]
    fn encode_ld_deref_c_a() {
        test_instruction(
            Ld(Special(SpecialLd::RegIndex, Direction::FromA)),
            bytes([0xe2]),
        )
    }

    #[test]
    fn encode_ld_a_deref_c() {
        test_instruction(
            Ld(Special(SpecialLd::RegIndex, Direction::IntoA)),
            bytes([0xf2]),
        )
    }

    #[test]
    fn encode_ldh() {
        let index = RelocExpr::Literal(0xcc, ());
        let test_cases = &[(Direction::FromA, 0xe0), (Direction::IntoA, 0xf0)];
        for &(direction, opcode) in test_cases {
            test_instruction(
                Ld(Special(SpecialLd::InlineIndex(index.clone()), direction)),
                [
                    DataItem::Byte(opcode),
                    DataItem::Expr(index.clone(), Width::Byte),
                ],
            )
        }
    }

    #[test]
    fn encode_alu_immediate() {
        use instruction::AluOperation::*;
        let expr = RelocExpr::Literal(0x42, ());
        [(Add, 0xc6), (And, 0xe6), (Cp, 0xfe), (Xor, 0xee)]
            .iter()
            .for_each(|(alu_operation, opcode)| {
                test_instruction(
                    Instruction::Alu(*alu_operation, AluSource::Immediate(expr.clone())),
                    [
                        DataItem::Byte(*opcode),
                        DataItem::Expr(expr.clone(), Width::Byte),
                    ],
                )
            })
    }

    #[test]
    fn encode_simple_add() {
        use instruction::SimpleOperand::*;
        let src_and_opcode = vec![
            (B, 0x80),
            (C, 0x81),
            (D, 0x82),
            (E, 0x83),
            (H, 0x84),
            (L, 0x85),
            (DerefHl, 0x86),
            (A, 0x87),
        ];
        test_simple_alu_encoding(AluOperation::Add, &src_and_opcode)
    }

    #[test]
    fn encode_simple_and() {
        use instruction::SimpleOperand::*;
        let src_and_opcode = vec![
            (B, 0xa0),
            (C, 0xa1),
            (D, 0xa2),
            (E, 0xa3),
            (H, 0xa4),
            (L, 0xa5),
            (DerefHl, 0xa6),
            (A, 0xa7),
        ];
        test_simple_alu_encoding(AluOperation::And, &src_and_opcode)
    }

    #[test]
    fn encode_simple_cp() {
        use instruction::SimpleOperand::*;
        let src_and_opcode = vec![
            (B, 0xb8),
            (C, 0xb9),
            (D, 0xba),
            (E, 0xbb),
            (H, 0xbc),
            (L, 0xbd),
            (DerefHl, 0xbe),
            (A, 0xbf),
        ];
        test_simple_alu_encoding(AluOperation::Cp, &src_and_opcode)
    }

    #[test]
    fn encode_simple_xor() {
        use instruction::SimpleOperand::*;
        let src_and_opcode = vec![
            (B, 0xa8),
            (C, 0xa9),
            (D, 0xaa),
            (E, 0xab),
            (H, 0xac),
            (L, 0xad),
            (DerefHl, 0xae),
            (A, 0xaf),
        ];
        test_simple_alu_encoding(AluOperation::Xor, &src_and_opcode)
    }

    fn test_simple_alu_encoding(operation: AluOperation, test_cases: &[(SimpleOperand, u8)]) {
        for &(src, opcode) in test_cases {
            test_instruction(Alu(operation, AluSource::Simple(src)), bytes([opcode]))
        }
    }

    #[test]
    fn encode_call() {
        use instruction::Condition::*;
        let target_expr = RelocExpr::Literal(0x1234, ());
        let test_cases = &[
            (None, 0xcd),
            (Some(C), 0xdc),
            (Some(Nc), 0xd4),
            (Some(Nz), 0xc4),
            (Some(Z), 0xcc),
        ];
        for &(condition, opcode) in test_cases {
            test_instruction(
                Branch(Call(target_expr.clone()), condition),
                [
                    DataItem::Byte(opcode),
                    DataItem::Expr(target_expr.clone(), Width::Word),
                ],
            )
        }
    }

    #[test]
    fn encode_jp() {
        use instruction::Condition::*;
        let target_expr = RelocExpr::Literal(0x1234, ());
        let test_cases = &[
            (None, 0xc3),
            (Some(C), 0xda),
            (Some(Nc), 0xd2),
            (Some(Nz), 0xc2),
            (Some(Z), 0xca),
        ];
        for &(condition, opcode) in test_cases {
            test_instruction(
                Branch(Jp(target_expr.clone()), condition),
                [
                    DataItem::Byte(opcode),
                    DataItem::Expr(target_expr.clone(), Width::Word),
                ],
            )
        }
    }

    #[test]
    fn encode_jp_hl() {
        test_instruction(JpDerefHl, bytes([0xe9]))
    }

    #[test]
    fn encode_jr() {
        use instruction::Condition::*;
        let target_expr = RelocExpr::Literal(0x1234, ());
        let test_cases = &[
            (None, 0x18),
            (Some(C), 0x38),
            (Some(Nc), 0x30),
            (Some(Nz), 0x20),
            (Some(Z), 0x28),
        ];
        for &(condition, opcode) in test_cases {
            test_instruction(
                Branch(Jr(target_expr.clone()), condition),
                [
                    DataItem::Byte(opcode),
                    DataItem::Expr(
                        RelocExpr::Subtract(
                            Box::new(target_expr.clone()),
                            Box::new(RelocExpr::LocationCounter),
                        ),
                        Width::Byte,
                    ),
                ],
            )
        }
    }

    #[test]
    fn encode_ret() {
        use instruction::Condition::*;
        let test_cases = &[
            (None, 0xc9),
            (Some(C), 0xd8),
            (Some(Nc), 0xd0),
            (Some(Nz), 0xc0),
            (Some(Z), 0xc8),
        ];
        for &(condition, opcode) in test_cases {
            test_instruction(Branch(Ret, condition), bytes([opcode]))
        }
    }

    #[test]
    fn encode_add_hl() {
        use instruction::Reg16::*;
        [(Bc, 0x09), (De, 0x19), (Hl, 0x29), (Sp, 0x39)]
            .iter()
            .for_each(|(reg16, opcode)| {
                test_instruction(Instruction::AddHl(*reg16), bytes([*opcode]))
            })
    }

    #[test]
    fn encode_inc_dec8() {
        use instruction::{IncDec::*, SimpleOperand::*};
        let test_cases = &[
            (Inc, B, 0x04),
            (Inc, C, 0x0c),
            (Inc, D, 0x14),
            (Inc, E, 0x1c),
            (Inc, H, 0x24),
            (Inc, L, 0x2c),
            (Inc, DerefHl, 0x34),
            (Inc, A, 0x3c),
            (Dec, B, 0x05),
            (Dec, C, 0x0d),
            (Dec, D, 0x15),
            (Dec, E, 0x1d),
            (Dec, H, 0x25),
            (Dec, L, 0x2d),
            (Dec, DerefHl, 0x35),
            (Dec, A, 0x3d),
        ];
        for (mode, operand, opcode) in test_cases {
            test_instruction(Instruction::IncDec8(*mode, *operand), bytes([*opcode]))
        }
    }

    #[test]
    fn encode_inc_dec16() {
        use instruction::{IncDec::*, Reg16::*};
        let test_cases = &[
            (Inc, Bc, 0x03),
            (Inc, De, 0x13),
            (Inc, Hl, 0x23),
            (Inc, Sp, 0x33),
            (Dec, Bc, 0x0b),
            (Dec, De, 0x1b),
            (Dec, Hl, 0x2b),
            (Dec, Sp, 0x3b),
        ];
        for (mode, operand, opcode) in test_cases {
            test_instruction(Instruction::IncDec16(*mode, *operand), bytes([*opcode]))
        }
    }

    #[test]
    fn encode_pop() {
        use instruction::RegPair::*;
        [(Bc, 0xc1), (De, 0xd1), (Hl, 0xe1), (Af, 0xf1)]
            .iter()
            .for_each(|(reg_pair, opcode)| {
                test_instruction(Instruction::Pop(*reg_pair), bytes([*opcode]))
            })
    }

    #[test]
    fn encode_push() {
        use instruction::RegPair::*;
        [(Bc, 0xc5), (De, 0xd5), (Hl, 0xe5), (Af, 0xf5)]
            .iter()
            .for_each(|(reg_pair, opcode)| {
                test_instruction(Instruction::Push(*reg_pair), bytes([*opcode]))
            })
    }

    #[test]
    fn encode_reti() {
        test_nullary(Reti, bytes([0xd9]))
    }

    fn bytes(data: impl Borrow<[u8]>) -> Vec<DataItem<()>> {
        data.borrow().iter().map(|&b| DataItem::Byte(b)).collect()
    }
}
