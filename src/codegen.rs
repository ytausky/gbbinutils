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

    #[test]
    fn encode_nop() {
        let ast = ast::EmitBytes::new("nop", &[]);
        let mut code = vec![];
        generate_code(&ast, |byte| code.push(byte));
        assert_eq!(code, [0x00])
    }

    #[test]
    fn encode_stop() {
        let ast = ast::EmitBytes::new("stop", &[]);
        let mut code = vec![];
        generate_code(&ast, |byte| code.push(byte));
        assert_eq!(code, [0x10, 0x00])
    }

    #[test]
    fn encode_halt() {
        let ast = ast::EmitBytes::new("halt", &[]);
        let mut code = vec![];
        generate_code(&ast, |byte| code.push(byte));
        assert_eq!(code, [0x76])
    }
}
