use backend::*;

pub trait Emit<R> {
    fn emit(&mut self, item: DataItem<R>);

    fn emit_byte(&mut self, value: u8) {
        self.emit(DataItem::Byte(value))
    }
}

#[derive(Debug, PartialEq)]
pub enum DataItem<R> {
    Byte(u8),
    Expr(Expr<R>, Width),
}

pub fn generate_code<R>(instruction: &Instruction<R>, emitter: &mut impl Emit<R>) {
    use backend::Instruction::*;
    match instruction {
        Halt => emitter.emit_byte(0x76),
        Ld(LdKind::Simple(dest, src)) => emitter.emit_byte(encode_ld_to_reg_from_reg(*dest, *src)),
        Nop => emitter.emit_byte(0x00),
        Stop => {
            emitter.emit_byte(0x10);
            emitter.emit_byte(0x00)
        }
        _ => panic!(),
    }
}

fn encode_ld_to_reg_from_reg(dest: SimpleOperand, src: SimpleOperand) -> u8 {
    0b01_000_000 | (encode_register(dest) << 3) | encode_register(src)
}

fn encode_register(register: SimpleOperand) -> u8 {
    use backend::SimpleOperand::*;
    match register {
        A => 0b111,
        B => 0b000,
        C => 0b001,
        D => 0b010,
        E => 0b011,
        H => 0b100,
        L => 0b101,
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use backend::Instruction::*;

    impl<F: FnMut(DataItem<()>)> Emit<()> for F {
        fn emit(&mut self, item: DataItem<()>) {
            self(item)
        }
    }

    fn test_instruction(instruction: Instruction<()>, bytes: &[u8]) {
        let mut code = vec![];
        generate_code(&instruction, &mut |item| code.push(item));
        assert_eq!(
            code,
            bytes.iter().map(|&b| DataItem::Byte(b)).collect::<Vec<_>>()
        )
    }

    #[test]
    fn encode_nop() {
        test_instruction(Nop, &[0x00])
    }

    #[test]
    fn encode_stop() {
        test_instruction(Stop, &[0x10, 0x00])
    }

    #[test]
    fn encode_halt() {
        test_instruction(Halt, &[0x76])
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
            test_instruction(Ld(LdKind::Simple(dest, src)), &[opcode])
        }
    }
}
