use ir;

use std;

mod ast;
mod semantics;
mod syntax;
mod token;

#[derive(Clone, Debug, PartialEq)]
pub enum Keyword {
    A,
    B,
    Bc,
    Endm,
    Halt,
    Hl,
    Include,
    Ld,
    Macro,
    Nop,
    Push,
    Stop,
    Xor,
}

pub fn analyze_file(name: &str) {
    use std::io::prelude::*;
    let mut file = std::fs::File::open(name).unwrap();
    let mut src = String::new();
    file.read_to_string(&mut src).unwrap();
    let mut ast_builder = semantics::AstBuilder::new(DumpSection::new());
    syntax::parse_src(
        syntax::lexer::Lexer::new(&src),
        &mut ast_builder,
        ast::ExprBuilder::new(),
    )
}

struct DumpSection;

impl DumpSection {
    fn new() -> DumpSection {
        DumpSection {}
    }
}

impl ir::Section for DumpSection {
    fn add_instruction(&mut self, instruction: ir::Instruction) {
        println!("{:?}", instruction)
    }

    fn add_label(&mut self, label: &str) {
        println!("Define symbol: {}", label)
    }
}
