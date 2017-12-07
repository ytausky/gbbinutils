#[cfg(test)]
use ast;

#[cfg(test)]
fn generate_code<F: FnMut(u8)>(ast_node: &ast::EmitBytes, mut sink: F) {
    match ast_node.mnemonic.as_ref() {
        "halt" => sink(0x76),
        "ld" => sink(encode_ld(ast_node.operands[1])),
        _ => {
            if ast_node.mnemonic == "stop" {
                sink(0x10)
            }
            sink(0x00)
        },
    }
}

#[cfg(test)]
fn encode_ld(src: ast::Operand) -> u8 {
    match src {
        ast::Operand::Register(register) => 0b01_111_000 | encode_register(register),
        _ => panic!(),
    }
}

#[cfg(test)]
fn encode_register(register: ast::Register) -> u8 {
    match register {
        ast::Register::A => 0b111,
        ast::Register::B => 0b000,
        ast::Register::C => 0b001,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_instruction(mnemonic: &str, operands: &[ast::Operand], bytes: &[u8]) {
        let ast = ast::EmitBytes::new(mnemonic, operands);
        let mut code = vec![];
        generate_code(&ast, |byte| code.push(byte));
        assert_eq!(code, bytes)
    }

    fn test_nullary_instruction(mnemonic: &str, bytes: &[u8]) {
        test_instruction(mnemonic, &[], bytes)
    }

    #[test]
    fn encode_nop() {
        test_nullary_instruction("nop", &[0x00])
    }

    #[test]
    fn encode_stop() {
        test_nullary_instruction("stop", &[0x10, 0x00])
    }

    #[test]
    fn encode_halt() {
        test_nullary_instruction("halt", &[0x76])
    }

    #[test]
    fn encode_ld_a_a() {
        test_instruction("ld", &[ast::A, ast::A], &[0x7f])
    }

    #[test]
    fn encode_ld_a_b() {
        test_instruction("ld", &[ast::A, ast::B], &[0x78])
    }

    #[test]
    fn encode_ld_a_c() {
        test_instruction("ld", &[ast::A, ast::C], &[0x79])
    }
}
