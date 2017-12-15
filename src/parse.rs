#[cfg(test)]
use ast;

#[cfg(test)]
use keyword;

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
    type Item = ast::AsmItem<'a>;

    fn next(&mut self) -> Option<ast::AsmItem<'a>> {
        let mut parsed_line = None;
        while parsed_line == None {
            parsed_line = parse_line(self.src.next()?)
        };
        parsed_line
    }
}

#[cfg(test)]
fn parse_line(line: &str) -> Option<ast::AsmItem> {
    let mut word_iterator = line.split_whitespace();
    if let Some(first_word) = word_iterator.next() {
        let first_mnemonic = parse_mnemonic(first_word);
        match first_mnemonic {
            keyword::Mnemonic::Include => Some(include(parse_include_path(word_iterator.next().unwrap()))),
            _ => Some(inst(first_mnemonic, &parse_operands(word_iterator))),
        }
    } else {
        None
    }
}

#[cfg(test)]
fn parse_mnemonic(spelling: &str) -> keyword::Mnemonic {
    match spelling {
        "halt" => keyword::Mnemonic::Halt,
        "include" => keyword::Mnemonic::Include,
        "ld" => keyword::Mnemonic::Ld,
        "nop" => keyword::Mnemonic::Nop,
        "push" => keyword::Mnemonic::Push,
        "stop" => keyword::Mnemonic::Stop,
        _ => unimplemented!(),
    }
}

#[cfg(test)]
fn parse_include_path(path: &str) -> &str {
    &path[1 .. path.len() - 1]
}

#[cfg(test)]
fn parse_operands<'a, I: Iterator<Item=&'a str>>(word_iterator: I) -> Vec<ast::Operand> {
    word_iterator.map(|op| parse_operand(op).unwrap()).collect()
}

#[cfg(test)]
fn parse_operand(src: &str) -> Option<ast::Operand> {
    let without_comma = if src.ends_with(',') {
        &src[0 .. src.len() - 1]
    } else {
        src
    };
    match without_comma {
        "a" => Some(ast::Operand::Register(keyword::Register::A)),
        "b" => Some(ast::Operand::Register(keyword::Register::B)),
        "bc" => Some(ast::Operand::RegisterPair(keyword::RegisterPair::Bc)),
        _ => None,
    }
}

#[cfg(test)]
fn inst<'a>(mnemonic: keyword::Mnemonic, operands: &[ast::Operand]) -> ast::AsmItem<'a> {
    ast::AsmItem::Instruction(ast::Instruction::new(mnemonic, operands))
}

#[cfg(test)]
fn include(path: &str) -> ast::AsmItem {
    ast::AsmItem::Include(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ast::*;

    use keyword::Mnemonic::*;

    fn assert_ast_eq(src: &str, expected_ast: &[AsmItem]) {
        let actual = parse_src(src).collect::<Vec<AsmItem>>();
        assert_eq!(actual, expected_ast)
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
        assert_ast_eq("    nop", &[inst(Nop, &[])])
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
        assert_ast_eq(src, &[inst(parse_mnemonic(src), &[])])
    }

    #[test]
    fn parse_push_bc() {
        assert_ast_eq("push bc", &[inst(Push, &[BC])])
    }

    #[test]
    fn parse_ld_a_a() {
        assert_ast_eq("ld a, a", &[inst(Ld, &[A, A])])
    }

    #[test]
    fn parse_ld_a_b() {
        assert_ast_eq("ld a, b", &[inst(Ld, &[A, B])])
    }

    #[test]
    fn parse_two_instructions() {
        assert_ast_eq("ld a, b\nld a, b", &[
            inst(Ld, &[A, B]),
            inst(Ld, &[A, B]),
        ])
    }

    #[test]
    fn parse_two_instructions_separated_by_blank_line() {
        assert_ast_eq("ld a, b\n\nld a, b", &[
            inst(Ld, &[A, B]),
            inst(Ld, &[A, B]),
        ])
    }

    #[test]
    fn parse_include() {
        assert_ast_eq("include \"file.asm\"", &[include("file.asm")])
    }
}
