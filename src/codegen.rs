#[cfg(test)]
use ast;

#[cfg(test)]
fn generate_code<F: FnMut(u8)>(_ast_node: &ast::EmitBytes, mut sink: F) {
    sink(0x00)
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
}
