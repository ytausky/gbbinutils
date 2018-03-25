mod ast;
mod codegen;
mod ir;
mod keyword;
mod lexer;
mod parse;
mod semantics;
mod syntax;
mod token;

pub fn analyze_file(name: &str) {
    use std::io::prelude::*;
    let mut file = std::fs::File::open(name).unwrap();
    let mut src = String::new();
    file.read_to_string(&mut src).unwrap();
    let mut ast_builder = semantics::AstBuilder::new(DumpSection::new());
    parse::parse_src(
        lexer::Lexer::new(&src),
        &mut ast_builder,
        ast::ExprBuilder::new(),
    );
    dump_ast(ast_builder.ast());
}

fn dump_ast(ast: &Vec<ast::AsmItem>) {
    for ref item in ast {
        println!("{:?}", item)
    }
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
