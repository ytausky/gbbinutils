#[cfg(test)]
use ast;

#[cfg(test)]
use std::str;

#[cfg(test)]
fn parse_src(src: &str) -> Parser {
    Parser {
        src: src.lines(),
    }
}

#[cfg(test)]
struct Parser<'a> {
    src: str::Lines<'a>,
}

#[cfg(test)]
impl<'a> Iterator for Parser<'a> {
    type Item = ast::EmitBytes;

    fn next(&mut self) -> Option<ast::EmitBytes> {
        let mut parsed_line = None;
        while parsed_line == None {
            parsed_line = parse_line(self.src.next()?)
        };
        parsed_line
    }
}

#[cfg(test)]
fn parse_line(line: &str) -> Option<ast::EmitBytes> {
    let trimmed_line = line.trim();
    if let Some(first_space) = trimmed_line.find(' ') {
        let (mnemonic, operands) = trimmed_line.split_at(first_space);
        Some(ast::EmitBytes::new(mnemonic, &parse_operands(operands)))
    } else {
        match trimmed_line {
            "nop" | "halt" | "stop" => Some(ast::EmitBytes::new(trimmed_line, &[])),
            _ => None
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

    #[test]
    fn parse_ld_a_a() {
        assert_ast_eq("ld a, a", &[("ld", &[ast::A, ast::A])])
    }

    #[test]
    fn parse_ld_a_b() {
        assert_ast_eq("ld a, b", &[("ld", &[ast::A, ast::B])])
    }

    #[test]
    fn parse_two_instructions() {
        assert_ast_eq("ld a, b\nld a, b", &[
            ("ld", &[ast::A, ast::B]),
            ("ld", &[ast::A, ast::B]),
        ])
    }

    #[test]
    fn parse_two_instructions_separated_by_blank_line() {
        assert_ast_eq("ld a, b\n\nld a, b", &[
            ("ld", &[ast::A, ast::B]),
            ("ld", &[ast::A, ast::B]),
        ])
    }
}
