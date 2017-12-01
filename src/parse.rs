#[cfg(test)]
use ast;

#[cfg(test)]
fn parse_src(src: &str) -> ast::AssemblyCommands {
    if src == "nop" {
        vec![make_emit_bytes("nop")]
    } else {
        vec![]
    }
}

#[cfg(test)]
fn make_emit_bytes(mnemonic: &str) -> ast::EmitBytes {
    ast::EmitBytes { mnemonic: mnemonic.to_owned(), }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_ast_eq(src: &str, mnemonics: &[&str]) {
        let expected_ast = mnemonics.iter()
                                    .map(|x| make_emit_bytes(*x))
                                    .collect::<Vec<ast::EmitBytes>>();
        assert_eq!(parse_src(src), expected_ast)
    }

    #[test]
    fn parse_empty_src() {
        assert_ast_eq("", &[])
    }

    #[test]
    fn parse_empty_line() {
        assert_ast_eq("\n", &[])
    }

    #[test]
    fn parse_nop() {
        assert_ast_eq("nop", &["nop"])
    }

    #[test]
    fn parse_nop_after_whitespace () {
        assert_ast_eq("    nop", &["nop"])
    }
}
