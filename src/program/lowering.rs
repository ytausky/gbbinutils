use super::{Node, RelocExpr};
use crate::expr::{BinaryOperator, ExprVariant};
use crate::model::*;
use crate::span::Source;
use std::mem;

pub(super) trait Lower<S> {
    fn lower(self) -> LoweredItem<S>;
}

pub(super) enum LoweredItem<S> {
    None,
    One(Node<S>),
    Two(Node<S>, Node<S>),
}

impl<S> Iterator for LoweredItem<S> {
    type Item = Node<S>;

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

impl<S> LoweredItem<S> {
    fn with_opcode(opcode: u8) -> LoweredItem<S> {
        LoweredItem::One(Node::Byte(opcode))
    }

    fn extended(opcode: impl Into<Node<S>>) -> LoweredItem<S> {
        LoweredItem::Two(Node::Byte(0xcb), opcode.into())
    }

    fn and_byte(self, expr: RelocExpr<S>) -> Self {
        self.and_expr(expr, Width::Byte)
    }

    fn and_word(self, expr: RelocExpr<S>) -> Self {
        self.and_expr(expr, Width::Word)
    }

    fn and_expr(self, expr: RelocExpr<S>, width: Width) -> Self {
        match self {
            LoweredItem::One(item) => LoweredItem::Two(item, Node::Expr(expr, width)),
            LoweredItem::None | LoweredItem::Two(..) => panic!(),
        }
    }
}

impl<S> From<u8> for Node<S> {
    fn from(byte: u8) -> Self {
        Node::Byte(byte)
    }
}

impl<S: Clone> Lower<S> for Item<RelocExpr<S>> {
    fn lower(self) -> LoweredItem<S> {
        match self {
            Item::Data(expr, width) => LoweredItem::One(Node::Expr(expr, width)),
            Item::Instruction(instruction) => instruction.lower(),
        }
    }
}

impl<S: Clone> Lower<S> for Instruction<RelocExpr<S>> {
    fn lower(self) -> LoweredItem<S> {
        use self::Instruction::*;
        match self {
            AddHl(reg16) => LoweredItem::with_opcode(0x09 | encode_reg16(reg16)),
            Alu(operation, AluSource::Simple(src)) => encode_simple_alu_operation(operation, src),
            Alu(operation, AluSource::Immediate(expr)) => {
                encode_immediate_alu_operation(operation, expr)
            }
            Bit(operation, expr, operand) => LoweredItem::extended(Node::Embedded(
                encode_bit_operation(operation) | encode_simple_operand(operand),
                expr,
            )),
            Branch(branch, condition) => encode_branch(branch, condition),
            IncDec8(mode, operand) => LoweredItem::with_opcode(
                0b00_000_100 | encode_inc_dec(mode) | (encode_simple_operand(operand) << 3),
            ),
            IncDec16(mode, operand) => {
                LoweredItem::with_opcode(0x03 | (encode_inc_dec(mode) << 3) | encode_reg16(operand))
            }
            JpDerefHl => LoweredItem::with_opcode(0xe9),
            Ld(ld) => ld.lower(),
            Ldhl(offset) => LoweredItem::with_opcode(0xf8).and_byte(offset),
            Misc(operation, operand) => {
                LoweredItem::extended(operation.encode() | operand.encode())
            }
            Nullary(nullary) => nullary.lower(),
            Pop(reg_pair) => LoweredItem::with_opcode(0xc1 | (encode_reg_pair(reg_pair) << 4)),
            Push(reg_pair) => LoweredItem::with_opcode(0xc5 | (encode_reg_pair(reg_pair) << 4)),
            Rst(expr) => LoweredItem::One(Node::Embedded(0b11_000_111, expr)),
        }
    }
}

impl<S> Lower<S> for Nullary {
    fn lower(self) -> LoweredItem<S> {
        use self::Nullary::*;
        let opcode = match self {
            Cpl => 0x2f,
            Daa => 0x27,
            Di => 0xf3,
            Ei => 0xfb,
            Halt => 0x76,
            Nop => 0x00,
            Rla => 0x17,
            Rlca => 0x07,
            Rra => 0x1f,
            Rrca => 0x0f,
            Stop => 0x10,
            Reti => 0xd9,
        };
        if self == Stop {
            LoweredItem::Two(Node::Byte(opcode), Node::Byte(0x00))
        } else {
            LoweredItem::with_opcode(opcode)
        }
    }
}

impl<S> Lower<S> for Ld<RelocExpr<S>> {
    fn lower(self) -> LoweredItem<S> {
        match self {
            Ld::Simple(dest, src) => encode_ld_to_reg_from_reg(dest, src),
            Ld::Special(special, direction) => encode_special_ld(special, direction),
            Ld::SpHl => LoweredItem::with_opcode(0xf9),
            Ld::Immediate8(dest, immediate) => {
                LoweredItem::with_opcode(0x06 | (encode_simple_operand(dest) << 3))
                    .and_byte(immediate)
            }
            Ld::Immediate16(dest, immediate) => {
                LoweredItem::with_opcode(0x01 | encode_reg16(dest)).and_word(immediate)
            }
        }
    }
}

fn encode_special_ld<S>(ld: SpecialLd<RelocExpr<S>>, direction: Direction) -> LoweredItem<S> {
    let direction_bit = encode_direction(direction);
    match ld {
        SpecialLd::DerefPtrReg(ptr_reg) => {
            LoweredItem::with_opcode(0x02 | encode_ptr_reg(ptr_reg) | (direction_bit >> 1))
        }
        SpecialLd::InlineAddr(addr) => {
            LoweredItem::One(Node::LdInlineAddr(0xe0 | direction_bit, addr))
        }
        SpecialLd::RegIndex => LoweredItem::with_opcode(0xe2 | direction_bit),
    }
}

fn encode_simple_alu_operation<S>(operation: AluOperation, src: SimpleOperand) -> LoweredItem<S> {
    LoweredItem::with_opcode(
        0b10_000_000 | encode_alu_operation(operation) | encode_simple_operand(src),
    )
}

fn encode_immediate_alu_operation<S>(
    operation: AluOperation,
    expr: RelocExpr<S>,
) -> LoweredItem<S> {
    LoweredItem::with_opcode(0b11_000_110 | encode_alu_operation(operation)).and_byte(expr)
}

fn encode_alu_operation(operation: AluOperation) -> u8 {
    use self::AluOperation::*;
    (match operation {
        Add => 0b000,
        Adc => 0b001,
        Sub => 0b010,
        Sbc => 0b011,
        And => 0b100,
        Xor => 0b101,
        Or => 0b110,
        Cp => 0b111,
    }) << 3
}

fn encode_branch<S: Clone>(
    branch: Branch<RelocExpr<S>>,
    condition: Option<Condition>,
) -> LoweredItem<S> {
    use self::Branch::*;
    match branch {
        Call(target) => LoweredItem::with_opcode(match condition {
            None => 0xcd,
            Some(condition) => 0xc4 | encode_condition(condition),
        })
        .and_word(target),
        Jp(target) => LoweredItem::with_opcode(match condition {
            None => 0xc3,
            Some(condition) => 0xc2 | encode_condition(condition),
        })
        .and_word(target),
        Jr(target) => LoweredItem::with_opcode(match condition {
            None => 0x18,
            Some(condition) => 0x20 | encode_condition(condition),
        })
        .and_byte(mk_relative_expr(target)),
        Ret => LoweredItem::with_opcode(match condition {
            None => 0xc9,
            Some(condition) => 0b11_000_000 | encode_condition(condition),
        }),
    }
}

fn mk_relative_expr<S: Clone>(expr: RelocExpr<S>) -> RelocExpr<S> {
    let span = expr.span();
    RelocExpr {
        variant: ExprVariant::Binary(
            BinaryOperator::Minus,
            Box::new(expr),
            Box::new(RelocExpr {
                variant: ExprVariant::Atom(RelocAtom::LocationCounter),
                span: span.clone(),
            }),
        ),
        span,
    }
}

fn encode_bit_operation(operation: BitOperation) -> u8 {
    use self::BitOperation::*;
    (match operation {
        Bit => 0b01,
        Set => 0b11,
        Res => 0b10,
    }) << 6
}

impl MiscOperation {
    fn encode(self) -> u8 {
        use self::MiscOperation::*;
        (match self {
            Rlc => 0b000,
            Rrc => 0b001,
            Rl => 0b010,
            Rr => 0b011,
            Sla => 0b100,
            Sra => 0b101,
            Swap => 0b110,
            Srl => 0b111,
        }) << 3
    }
}

fn encode_condition(condition: Condition) -> u8 {
    use self::Condition::*;
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

fn encode_ld_to_reg_from_reg<S>(dest: SimpleOperand, src: SimpleOperand) -> LoweredItem<S> {
    LoweredItem::with_opcode(
        0b01_000_000 | (encode_simple_operand(dest) << 3) | encode_simple_operand(src),
    )
}

fn encode_simple_operand(operand: SimpleOperand) -> u8 {
    operand.encode()
}

impl SimpleOperand {
    fn encode(self) -> u8 {
        use self::SimpleOperand::*;
        match self {
            B => 0b000,
            C => 0b001,
            D => 0b010,
            E => 0b011,
            H => 0b100,
            L => 0b101,
            DerefHl => 0b110,
            A => 0b111,
        }
    }
}

fn encode_reg16(reg16: Reg16) -> u8 {
    use self::Reg16::*;
    (match reg16 {
        Bc => 0b00,
        De => 0b01,
        Hl => 0b10,
        Sp => 0b11,
    }) << 4
}

fn encode_reg_pair(reg_pair: RegPair) -> u8 {
    use self::RegPair::*;
    match reg_pair {
        Bc => 0b00,
        De => 0b01,
        Hl => 0b10,
        Af => 0b11,
    }
}

fn encode_ptr_reg(ptr_reg: PtrReg) -> u8 {
    use self::PtrReg::*;
    (match ptr_reg {
        Bc => 0b00,
        De => 0b01,
        Hli => 0b10,
        Hld => 0b11,
    }) << 4
}

fn encode_inc_dec(mode: IncDec) -> u8 {
    use self::IncDec::*;
    match mode {
        Inc => 0,
        Dec => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Borrow;

    use crate::model::{self, Branch::*, Instruction::*, Ld::*, Nullary::*};

    fn test_instruction(
        instruction: Instruction<RelocExpr<()>>,
        data_items: impl Borrow<[Node<()>]>,
    ) {
        let code: Vec<_> = instruction.lower().collect();
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

    #[test]
    fn lower_rlca() {
        test_nullary(Rlca, bytes([0x07]))
    }

    #[test]
    fn lower_rrca() {
        test_nullary(Rrca, bytes([0x0f]))
    }

    #[test]
    fn lower_rla() {
        test_nullary(Rla, bytes([0x17]))
    }

    #[test]
    fn lower_rra() {
        test_nullary(Rra, bytes([0x1f]))
    }

    #[test]
    fn lower_cpl() {
        test_nullary(Cpl, bytes([0x2f]))
    }

    fn test_nullary(nullary: model::Nullary, items: impl Borrow<[Node<()>]>) {
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
        use self::SimpleOperand::*;
        let immediate: RelocExpr<_> = 0x42.into();
        vec![
            (B, 0x06),
            (C, 0x0e),
            (D, 0x16),
            (E, 0x1e),
            (H, 0x26),
            (L, 0x2e),
            (DerefHl, 0x36),
            (A, 0x3e),
        ]
        .into_iter()
        .for_each(|(dest, opcode)| {
            test_instruction(
                Ld(Immediate8(dest, immediate.clone())),
                [
                    Node::Byte(opcode),
                    Node::Expr(immediate.clone(), Width::Byte),
                ],
            )
        })
    }

    #[test]
    fn encode_ld_immediate16() {
        use self::Reg16::*;
        let immediate: RelocExpr<_> = 0x1234.into();
        let test_cases = &[(Bc, 0x01), (De, 0x11), (Hl, 0x21), (Sp, 0x31)];
        for &(reg16, opcode) in test_cases {
            test_instruction(
                Ld(Immediate16(reg16, immediate.clone())),
                [
                    Node::Byte(opcode),
                    Node::Expr(immediate.clone(), Width::Word),
                ],
            )
        }
    }

    #[test]
    fn encode_ld_inline_addr() {
        let addr: RelocExpr<_> = 0x1234.into();
        let test_cases = &[(Direction::FromA, 0xea), (Direction::IntoA, 0xfa)];
        for &(direction, opcode) in test_cases {
            test_instruction(
                Ld(Special(SpecialLd::InlineAddr(addr.clone()), direction)),
                [Node::LdInlineAddr(opcode & 0xf0, addr.clone())],
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
    fn lower_ld_deref_ptr_reg() {
        use self::{Direction::*, PtrReg::*};
        let test_cases = &[
            (Bc, FromA, 0x02),
            (De, FromA, 0x12),
            (Hli, FromA, 0x22),
            (Hld, FromA, 0x32),
            (Bc, IntoA, 0x0a),
            (De, IntoA, 0x1a),
            (Hli, IntoA, 0x2a),
            (Hld, IntoA, 0x3a),
        ];
        for &(ptr_reg, direction, opcode) in test_cases {
            test_instruction(
                Ld(Special(SpecialLd::DerefPtrReg(ptr_reg), direction)),
                bytes([opcode]),
            )
        }
    }

    #[test]
    fn lower_ld_sp_hl() {
        test_instruction(Ld(SpHl), bytes([0xf9]))
    }

    #[test]
    fn lower_ldhl_sp_expr() {
        let expr: RelocExpr<_> = 0x42.into();
        test_instruction(
            Ldhl(expr.clone()),
            [Node::Byte(0xf8), Node::Expr(expr, Width::Byte)],
        )
    }

    #[test]
    fn encode_alu_immediate() {
        use self::AluOperation::*;
        let expr: RelocExpr<_> = 0x42.into();
        [
            (Add, 0xc6),
            (Adc, 0xce),
            (Sub, 0xd6),
            (Sbc, 0xde),
            (And, 0xe6),
            (Xor, 0xee),
            (Or, 0xf6),
            (Cp, 0xfe),
        ]
        .iter()
        .for_each(|(alu_operation, opcode)| {
            test_instruction(
                Instruction::Alu(*alu_operation, AluSource::Immediate(expr.clone())),
                [Node::Byte(*opcode), Node::Expr(expr.clone(), Width::Byte)],
            )
        })
    }

    #[test]
    fn encode_simple_add() {
        use self::SimpleOperand::*;
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
    fn encode_simple_adc() {
        use self::SimpleOperand::*;
        let src_and_opcode = vec![
            (B, 0x88),
            (C, 0x89),
            (D, 0x8a),
            (E, 0x8b),
            (H, 0x8c),
            (L, 0x8d),
            (DerefHl, 0x8e),
            (A, 0x8f),
        ];
        test_simple_alu_encoding(AluOperation::Adc, &src_and_opcode)
    }

    #[test]
    fn lower_simple_sub() {
        use self::SimpleOperand::*;
        let src_and_opcode = vec![
            (B, 0x90),
            (C, 0x91),
            (D, 0x92),
            (E, 0x93),
            (H, 0x94),
            (L, 0x95),
            (DerefHl, 0x96),
            (A, 0x97),
        ];
        test_simple_alu_encoding(AluOperation::Sub, &src_and_opcode)
    }

    #[test]
    fn encode_simple_sbc() {
        use self::SimpleOperand::*;
        let src_and_opcode = vec![
            (B, 0x98),
            (C, 0x99),
            (D, 0x9a),
            (E, 0x9b),
            (H, 0x9c),
            (L, 0x9d),
            (DerefHl, 0x9e),
            (A, 0x9f),
        ];
        test_simple_alu_encoding(AluOperation::Sbc, &src_and_opcode)
    }

    #[test]
    fn encode_simple_and() {
        use self::SimpleOperand::*;
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
    fn encode_simple_xor() {
        use self::SimpleOperand::*;
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

    #[test]
    fn encode_simple_or() {
        use self::SimpleOperand::*;
        let src_and_opcode = vec![
            (B, 0xb0),
            (C, 0xb1),
            (D, 0xb2),
            (E, 0xb3),
            (H, 0xb4),
            (L, 0xb5),
            (DerefHl, 0xb6),
            (A, 0xb7),
        ];
        test_simple_alu_encoding(AluOperation::Or, &src_and_opcode)
    }

    #[test]
    fn encode_simple_cp() {
        use self::SimpleOperand::*;
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

    fn test_simple_alu_encoding(operation: AluOperation, test_cases: &[(SimpleOperand, u8)]) {
        for &(src, opcode) in test_cases {
            test_instruction(Alu(operation, AluSource::Simple(src)), bytes([opcode]))
        }
    }

    #[test]
    fn encode_call() {
        use self::Condition::*;
        let target_expr: RelocExpr<_> = 0x1234.into();
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
                    Node::Byte(opcode),
                    Node::Expr(target_expr.clone(), Width::Word),
                ],
            )
        }
    }

    #[test]
    fn encode_jp() {
        use self::Condition::*;
        let target_expr: RelocExpr<_> = 0x1234.into();
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
                    Node::Byte(opcode),
                    Node::Expr(target_expr.clone(), Width::Word),
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
        use self::Condition::*;
        let target_expr: RelocExpr<_> = 0x1234.into();
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
                    Node::Byte(opcode),
                    Node::Expr(mk_relative_expr(target_expr.clone()), Width::Byte),
                ],
            )
        }
    }

    #[test]
    fn encode_ret() {
        use self::Condition::*;
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
        use self::Reg16::*;
        [(Bc, 0x09), (De, 0x19), (Hl, 0x29), (Sp, 0x39)]
            .iter()
            .for_each(|(reg16, opcode)| {
                test_instruction(Instruction::AddHl(*reg16), bytes([*opcode]))
            })
    }

    #[test]
    fn encode_inc_dec8() {
        use self::{IncDec::*, SimpleOperand::*};
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
        use self::{IncDec::*, Reg16::*};
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
        use self::RegPair::*;
        [(Bc, 0xc1), (De, 0xd1), (Hl, 0xe1), (Af, 0xf1)]
            .iter()
            .for_each(|(reg_pair, opcode)| {
                test_instruction(Instruction::Pop(*reg_pair), bytes([*opcode]))
            })
    }

    #[test]
    fn encode_push() {
        use self::RegPair::*;
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

    #[test]
    fn lower_rst() {
        let n: RelocExpr<_> = 3.into();
        test_instruction(
            Instruction::Rst(n.clone()),
            [Node::Embedded(0b11_000_111, n)],
        )
    }

    #[test]
    fn lower_bit_operations() {
        use self::{BitOperation::*, SimpleOperand::*};
        let n: RelocExpr<_> = 3.into();
        let test_cases = &[
            (Bit, B, 0b01_000_000),
            (Bit, C, 0b01_000_001),
            (Bit, D, 0b01_000_010),
            (Bit, E, 0b01_000_011),
            (Bit, H, 0b01_000_100),
            (Bit, L, 0b01_000_101),
            (Bit, DerefHl, 0b01_000_110),
            (Bit, A, 0b01_000_111),
            (Set, B, 0b11_000_000),
            (Set, C, 0b11_000_001),
            (Set, D, 0b11_000_010),
            (Set, E, 0b11_000_011),
            (Set, H, 0b11_000_100),
            (Set, L, 0b11_000_101),
            (Set, DerefHl, 0b11_000_110),
            (Set, A, 0b11_000_111),
            (Res, B, 0b10_000_000),
            (Res, C, 0b10_000_001),
            (Res, D, 0b10_000_010),
            (Res, E, 0b10_000_011),
            (Res, H, 0b10_000_100),
            (Res, L, 0b10_000_101),
            (Res, DerefHl, 0b10_000_110),
            (Res, A, 0b10_000_111),
        ];
        for &(operation, operand, opcode) in test_cases {
            test_instruction(
                Instruction::Bit(operation, n.clone(), operand),
                extended(Node::Embedded(opcode, n.clone())),
            )
        }
    }

    #[test]
    fn lower_misc_operations() {
        use self::{MiscOperation::*, SimpleOperand::*};
        let test_cases = &[
            (Rlc, H, 0x04),
            (Rrc, B, 0x08),
            (Rl, A, 0x17),
            (Rr, D, 0x1a),
            (Sla, C, 0x21),
            (Sra, E, 0x2b),
            (Swap, DerefHl, 0x36),
            (Srl, L, 0x3d),
        ];
        for &(operation, operand, opcode) in test_cases {
            test_instruction(Instruction::Misc(operation, operand), extended(opcode))
        }
    }

    fn extended(suffix: impl Into<Node<()>>) -> Vec<Node<()>> {
        vec![Node::Byte(0xcb), suffix.into()]
    }

    fn bytes(data: impl Borrow<[u8]>) -> Vec<Node<()>> {
        data.borrow().iter().map(|&b| Node::Byte(b)).collect()
    }
}
