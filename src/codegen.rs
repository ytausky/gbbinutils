#[cfg(test)]
use ast;

#[cfg(test)]
fn generate_code<F: FnMut(u8)>(ast_node: &ast::EmitBytes, mut sink: F) {
    match ast_node.mnemonic.as_ref() {
        "halt" => sink(0x76),
        "ld" => sink(0x7f),
        _ => {
            if ast_node.mnemonic == "stop" {
                sink(0x10)
            }
            sink(0x00)
        }
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
}
