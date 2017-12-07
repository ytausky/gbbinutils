#[cfg(test)]
use ast;

#[cfg(test)]
fn parse_src(src: &str) -> Parser {
    Parser {
        src,
    }
}

#[cfg(test)]
struct Parser<'a> {
    src: &'a str,
}

#[cfg(test)]
impl<'a> Iterator for Parser<'a> {
    type Item = ast::EmitBytes;

    fn next(&mut self) -> Option<ast::EmitBytes> {
        let trimmed_src = self.src.trim();
        self.src = &"";
        if let Some(first_space) = trimmed_src.find(' ') {
            let (mnemonic, operands) = trimmed_src.split_at(first_space);
            Some(ast::EmitBytes::new(mnemonic, &parse_operands(operands)))
        } else {
            match trimmed_src {
                "nop" | "halt" | "stop" => Some(ast::EmitBytes::new(trimmed_src, &[])),
                _ => None
            }
        }
    }
}

#[cfg(test)]
fn parse_operands(src: &str) -> Vec<ast::Operand> {
    src.split(',').map(|op| parse_operand(op).unwrap()).collect()
}

#[cfg(test)]
fn parse_operand(src: &str) -> Option<ast::Operand> {
    match src.trim() {
        "a" => Some(ast::Operand::Register(ast::Register::A)),
        "b" => Some(ast::Operand::Register(ast::Register::B)),
        "bc" => Some(ast::Operand::RegisterPair(ast::RegisterPair::Bc)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Command<'a> = (&'a str, &'a[ast::Operand]);

    fn make_ast(commands: &[Command]) -> ast::AssemblyCommands {
        commands.iter()
                .map(|&(mnemonic, operands)| ast::EmitBytes::new(mnemonic, operands))
                .collect()
    }

    fn assert_ast_eq(src: &str, commands: &[(&str, &[ast::Operand])]) {
        let actual = parse_src(src).collect::<Vec<ast::EmitBytes>>();
        let expected = make_ast(commands);
        assert_eq!(actual, expected)
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
        assert_ast_eq("    nop", &[("nop", &[])])
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
        assert_ast_eq(src, &[(src, &[])])
    }

    const BC: ast::Operand = ast::Operand::RegisterPair(ast::RegisterPair::Bc);

    #[test]
    fn parse_push_bc() {
        assert_ast_eq("push bc", &[("push", &[BC])])
    }

    const A: ast::Operand = ast::Operand::Register(ast::Register::A);
    const B: ast::Operand = ast::Operand::Register(ast::Register::B);

    #[test]
    fn parse_ld_a_a() {
        assert_ast_eq("ld a, a", &[("ld", &[A, A])])
    }

    #[test]
    fn parse_ld_a_b() {
        assert_ast_eq("ld a, b", &[("ld", &[A, B])])
    }
}
