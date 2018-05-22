use backend::*;

pub trait Emit<R> {
    fn emit(&mut self, item: DataItem<R>);

    fn emit_byte(&mut self, value: u8) {
        self.emit(DataItem::Byte(value))
    }

    fn emit_byte_expr(&mut self, value: Expr<R>) {
        self.emit(DataItem::Expr(value, Width::Byte))
    }

    fn emit_word(&mut self, value: Expr<R>) {
        self.emit(DataItem::Expr(value, Width::Word))
    }
}

#[derive(Debug, PartialEq)]
pub enum DataItem<R> {
    Byte(u8),
    Expr(Expr<R>, Width),
}

pub fn generate_code<R>(instruction: Instruction<R>, emitter: &mut impl Emit<R>) {
    use backend::Instruction::*;
    match instruction {
        AddHl(reg16) => emitter.emit_byte(0x09 | (encode_reg16(reg16) << 4)),
        Alu(operation, AluSource::Simple(src)) => {
            encode_simple_alu_operation(operation, src, emitter)
        }
        Branch(branch, condition) => encode_branch(branch, condition, emitter),
        Dec16(reg16) => emitter.emit_byte(0x0b | (encode_reg16(reg16) << 4)),
        Halt => emitter.emit_byte(0x76),
        Ld(kind) => encode_ld(kind, emitter),
        Nop => emitter.emit_byte(0x00),
        Stop => {
            emitter.emit_byte(0x10);
            emitter.emit_byte(0x00)
        }
        _ => panic!(),
    }
}

fn encode_ld<R>(kind: LdKind<R>, emitter: &mut impl Emit<R>) {
    match kind {
        LdKind::Simple(dest, src) => emitter.emit_byte(encode_ld_to_reg_from_reg(dest, src)),
        LdKind::Immediate8(dest, immediate) => {
            emitter.emit_byte(0x06 | (encode_simple_operand(dest) << 3));
            emitter.emit_byte_expr(immediate)
        }
        _ => panic!(),
    }
}

fn encode_simple_alu_operation<R>(
    operation: AluOperation,
    src: SimpleOperand,
    emitter: &mut impl Emit<R>,
) {
    match operation {
        AluOperation::Add => emitter.emit_byte(0x80 | encode_simple_operand(src)),
        _ => panic!(),
    }
}

fn encode_branch<R>(branch: Branch<R>, condition: Option<Condition>, emitter: &mut impl Emit<R>) {
    use backend::Branch::*;
    match (branch, condition) {
        (Jp(target), None) => {
            emitter.emit_byte(0xc3);
            emitter.emit_word(target)
        }
        _ => panic!(),
    }
}

fn encode_ld_to_reg_from_reg(dest: SimpleOperand, src: SimpleOperand) -> u8 {
    0b01_000_000 | (encode_simple_operand(dest) << 3) | encode_simple_operand(src)
}

fn encode_simple_operand(register: SimpleOperand) -> u8 {
    use backend::SimpleOperand::*;
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
    use backend::Reg16::*;
    match reg16 {
        Bc => 0b00,
        De => 0b01,
        Hl => 0b10,
        Sp => 0b11,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Borrow;

    use backend::Branch::*;
    use backend::Instruction::*;

    impl<F: FnMut(DataItem<()>)> Emit<()> for F {
        fn emit(&mut self, item: DataItem<()>) {
            self(item)
        }
    }

    fn test_instruction(instruction: Instruction<()>, data_items: impl Borrow<[DataItem<()>]>) {
        let mut code = vec![];
        generate_code(instruction, &mut |item| code.push(item));
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
        use backend::SimpleOperand::*;
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
    fn encode_simple_add() {
        use backend::SimpleOperand::*;
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
        for (src, opcode) in src_and_opcode {
            test_instruction(
                Alu(AluOperation::Add, AluSource::Simple(src)),
                bytes([opcode]),
            )
        }
    }

    #[test]
    fn encode_jp() {
        let target_expr = Expr::Literal(0x1234, ());
        test_instruction(
            Branch(Jp(target_expr.clone()), None),
            [
                DataItem::Byte(0xc3),
                DataItem::Expr(target_expr, Width::Word),
            ],
        )
    }

    #[test]
    fn encode_add_hl() {
        use backend::Reg16::*;
        [(Bc, 0x09), (De, 0x19), (Hl, 0x29), (Sp, 0x39)]
            .iter()
            .for_each(|(reg16, opcode)| {
                test_instruction(Instruction::AddHl(*reg16), bytes([*opcode]))
            })
    }

    #[test]
    fn encode_dec16() {
        use backend::Reg16::*;
        [(Bc, 0x0b), (De, 0x1b), (Hl, 0x2b), (Sp, 0x3b)]
            .iter()
            .for_each(|(reg16, opcode)| {
                test_instruction(Instruction::Dec16(*reg16), bytes([*opcode]))
            })
    }

    fn bytes(data: impl Borrow<[u8]>) -> Vec<DataItem<()>> {
        data.borrow().iter().map(|&b| DataItem::Byte(b)).collect()
    }
}
