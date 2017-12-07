#[cfg(test)]
use ast;

#[cfg(test)]
fn generate_code<F: FnMut(u8)>(ast_node: &ast::EmitBytes, mut sink: F) {
    if ast_node.mnemonic == "halt" {
        sink(0x76)
    } else {
        if ast_node.mnemonic == "stop" {
            sink(0x10)
        };
        sink(0x00)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_nullary_instruction(mnemonic: &str, bytes: &[u8]) {
        let ast = ast::EmitBytes::new(mnemonic, &[]);
        let mut code = vec![];
        generate_code(&ast, |byte| code.push(byte));
        assert_eq!(code, bytes)
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
}
