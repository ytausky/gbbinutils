#[cfg(test)]
use ast;

#[cfg(test)]
fn parse_src(src: &str) -> ast::AssemblyCommands {
    let trimmed_src = src.trim();
    match trimmed_src {
        "nop" => vec![make_emit_bytes("nop")],
        "halt" => vec![make_emit_bytes("halt")],
        "stop" => vec![make_emit_bytes("stop")],
        _ => vec![]
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
        parse_nullary_instruction("nop")
    }

    #[test]
    fn parse_nop_after_whitespace() {
        assert_ast_eq("    nop", &["nop"])
    }

    #[test]
    fn parse_halt() {
        parse_nullary_instruction("halt")
    }

    #[test]
    fn parse_stop() {
        parse_nullary_instruction("stop")
    }

    fn parse_nullary_instruction(src: &str) {
        assert_ast_eq(src, &[src])
    }
}
