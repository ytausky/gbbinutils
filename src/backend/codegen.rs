use Width;
use instruction::*;

pub trait Emit<R> {
    fn emit(&mut self, item: DataItem<R>);

    fn emit_encoded(&mut self, encoded: Encoded<R>) {
        self.emit(DataItem::Byte(encoded.opcode));
        if let Some(immediate) = encoded.immediate {
            self.emit(immediate)
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum DataItem<R> {
    Byte(u8),
    Expr(Expr<R>, Width),
}

pub struct Encoded<R> {
    opcode: u8,
    immediate: Option<DataItem<R>>,
}

impl<R> Encoded<R> {
    fn with(opcode: u8) -> Encoded<R> {
        Encoded {
            opcode,
            immediate: None,
        }
    }

    fn and_byte(mut self, expr: Expr<R>) -> Encoded<R> {
        self.immediate = Some(DataItem::Expr(expr, Width::Byte));
        self
    }

    fn and_word(mut self, expr: Expr<R>) -> Encoded<R> {
        self.immediate = Some(DataItem::Expr(expr, Width::Word));
        self
    }
}

pub fn generate_code<R>(instruction: Instruction<R>) -> Encoded<R> {
    use backend::Instruction::*;
    match instruction {
        AddHl(reg16) => Encoded::with(0x09 | encode_reg16(reg16)),
        Alu(operation, AluSource::Simple(src)) => encode_simple_alu_operation(operation, src),
        Alu(operation, AluSource::Immediate(expr)) => {
            encode_immediate_alu_operation(operation, expr)
        }
        Branch(branch, condition) => encode_branch(branch, condition),
        Dec8(simple_operand) => Encoded::with(0x05 | (encode_simple_operand(simple_operand) << 3)),
        Dec16(reg16) => Encoded::with(0x0b | encode_reg16(reg16)),
        Halt => Encoded::with(0x76),
        Inc8(operand) => Encoded::with(0b00_000_100 | (encode_simple_operand(operand) << 3)),
        Inc16(reg16) => Encoded::with(0x03 | encode_reg16(reg16)),
        JpDerefHl => Encoded::with(0xe9),
        Ld(kind) => encode_ld(kind),
        Ldh(expr, direction) => Encoded::with(0xe0 | encode_direction(direction)).and_byte(expr),
        Nop => Encoded::with(0x00),
        Stop => Encoded {
            opcode: 0x10,
            immediate: Some(DataItem::Byte(0x00)),
        },
        Pop(reg_pair) => Encoded::with(0xc1 | (encode_reg_pair(reg_pair) << 4)),
        Push(reg_pair) => Encoded::with(0xc5 | (encode_reg_pair(reg_pair) << 4)),
        Reti => Encoded::with(0xd9),
    }
}

fn encode_ld<R>(kind: LdKind<R>) -> Encoded<R> {
    match kind {
        LdKind::Simple(dest, src) => encode_ld_to_reg_from_reg(dest, src),
        LdKind::Immediate8(dest, immediate) => {
            Encoded::with(0x06 | (encode_simple_operand(dest) << 3)).and_byte(immediate)
        }
        LdKind::Immediate16(dest, immediate) => {
            Encoded::with(0x01 | encode_reg16(dest)).and_word(immediate)
        }
        LdKind::ImmediateAddr(addr, direction) => {
            Encoded::with(0xea | encode_direction(direction)).and_word(addr)
        }
    }
}

fn encode_simple_alu_operation<R>(operation: AluOperation, src: SimpleOperand) -> Encoded<R> {
    use instruction::AluOperation::*;
    let opcode_base = match operation {
        Add => 0x80,
        And => 0xa0,
        Cp => 0xb8,
        Xor => 0xa8,
    };
    Encoded::with(opcode_base | encode_simple_operand(src))
}

fn encode_immediate_alu_operation<R>(operation: AluOperation, expr: Expr<R>) -> Encoded<R> {
    use instruction::AluOperation::*;
    Encoded::with(match operation {
        Add => 0xc6,
        And => 0xe6,
        Cp => 0xfe,
        Xor => 0xee,
    }).and_byte(expr)
}

fn encode_branch<R>(branch: Branch<R>, condition: Option<Condition>) -> Encoded<R> {
    use instruction::Branch::*;
    match branch {
        Call(target) => Encoded::with(match condition {
            None => 0xcd,
            Some(condition) => 0xc4 | encode_condition(condition),
        }).and_word(target),
        Jp(target) => Encoded::with(match condition {
            None => 0xc3,
            Some(condition) => 0xc2 | encode_condition(condition),
        }).and_word(target),
        Jr(target) => Encoded::with(match condition {
            None => 0x18,
            Some(condition) => 0x20 | encode_condition(condition),
        }).and_byte(Expr::Subtract(
            Box::new(target),
            Box::new(Expr::LocationCounter),
        )),
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

fn encode_ld_to_reg_from_reg<R>(dest: SimpleOperand, src: SimpleOperand) -> Encoded<R> {
    Encoded::with(0b01_000_000 | (encode_simple_operand(dest) << 3) | encode_simple_operand(src))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Borrow;

    use instruction::Branch::*;
    use instruction::Instruction::*;

    impl<F: FnMut(DataItem<()>)> Emit<()> for F {
        fn emit(&mut self, item: DataItem<()>) {
            self(item)
        }
    }

    fn test_instruction(instruction: Instruction<()>, data_items: impl Borrow<[DataItem<()>]>) {
        let mut code = vec![];
        (&mut |item| code.push(item)).emit_encoded(generate_code(instruction));
        assert_eq!(code, data_items.borrow())
    }

    #[test]
    fn encode_nop() {
        test_instruction(Nop, bytes([0x00]))
    }

    #[test]
    fn encode_stop() {
        test_instruction(Stop, bytes([0x10, 0x00]))
    }

    #[test]
    fn encode_halt() {
        test_instruction(Halt, bytes([0x76]))
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
            test_instruction(Ld(LdKind::Simple(dest, src)), bytes([opcode]))
        }
    }

    #[test]
    fn encode_ld_simple_immediate() {
        use instruction::SimpleOperand::*;
        let immediate = Expr::Literal(0x42, ());
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
                    Ld(LdKind::Immediate8(dest, immediate.clone())),
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
        let immediate = Expr::Literal(0x1234, ());
        let test_cases = &[(Bc, 0x01), (De, 0x11), (Hl, 0x21), (Sp, 0x31)];
        for &(reg16, opcode) in test_cases {
            test_instruction(
                Ld(LdKind::Immediate16(reg16, immediate.clone())),
                [
                    DataItem::Byte(opcode),
                    DataItem::Expr(immediate.clone(), Width::Word),
                ],
            )
        }
    }

    #[test]
    fn encode_ld_immediate_addr() {
        let addr = Expr::Literal(0x1234, ());
        let test_cases = &[(Direction::FromA, 0xea), (Direction::IntoA, 0xfa)];
        for &(direction, opcode) in test_cases {
            test_instruction(
                Ld(LdKind::ImmediateAddr(addr.clone(), direction)),
                [
                    DataItem::Byte(opcode),
                    DataItem::Expr(addr.clone(), Width::Word),
                ],
            )
        }
    }

    #[test]
    fn encode_ldh() {
        let index = Expr::Literal(0xcc, ());
        let test_cases = &[(Direction::FromA, 0xe0), (Direction::IntoA, 0xf0)];
        for &(direction, opcode) in test_cases {
            test_instruction(
                Ldh(index.clone(), direction),
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
        let expr = Expr::Literal(0x42, ());
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
        let target_expr = Expr::Literal(0x1234, ());
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
        let target_expr = Expr::Literal(0x1234, ());
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
        let target_expr = Expr::Literal(0x1234, ());
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
                        Expr::Subtract(
                            Box::new(target_expr.clone()),
                            Box::new(Expr::LocationCounter),
                        ),
                        Width::Byte,
                    ),
                ],
            )
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
    fn encode_inc8() {
        use instruction::SimpleOperand::*;
        let test_cases = &[
            (B, 0x04),
            (C, 0x0c),
            (D, 0x14),
            (E, 0x1c),
            (H, 0x24),
            (L, 0x2c),
            (DerefHl, 0x34),
            (A, 0x3c),
        ];
        for (operand, opcode) in test_cases {
            test_instruction(Instruction::Inc8(*operand), bytes([*opcode]))
        }
    }

    #[test]
    fn encode_dec8() {
        use instruction::SimpleOperand::*;
        [
            (B, 0x05),
            (C, 0x0d),
            (D, 0x15),
            (E, 0x1d),
            (H, 0x25),
            (L, 0x2d),
            (DerefHl, 0x35),
            (A, 0x3d),
        ].iter()
            .for_each(|(simple_operand, opcode)| {
                test_instruction(Instruction::Dec8(*simple_operand), bytes([*opcode]))
            })
    }

    #[test]
    fn encode_inc16() {
        use instruction::Reg16::*;
        [(Bc, 0x03), (De, 0x13), (Hl, 0x23), (Sp, 0x33)]
            .iter()
            .for_each(|(reg16, opcode)| {
                test_instruction(Instruction::Inc16(*reg16), bytes([*opcode]))
            })
    }

    #[test]
    fn encode_dec16() {
        use instruction::Reg16::*;
        [(Bc, 0x0b), (De, 0x1b), (Hl, 0x2b), (Sp, 0x3b)]
            .iter()
            .for_each(|(reg16, opcode)| {
                test_instruction(Instruction::Dec16(*reg16), bytes([*opcode]))
            })
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
        test_instruction(Instruction::Reti, bytes([0xd9]))
    }

    fn bytes(data: impl Borrow<[u8]>) -> Vec<DataItem<()>> {
        data.borrow().iter().map(|&b| DataItem::Byte(b)).collect()
    }
}
