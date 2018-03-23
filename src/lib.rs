mod ast;
mod codegen;
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
    let mut ast_builder = semantics::AstBuilder::new();
    parse::parse_src(lexer::Lexer::new(&src), &mut ast_builder);
    dump_ast(ast_builder.ast());
}

fn dump_ast(ast: &Vec<ast::AsmItem>) {
    for ref item in ast {
        println!("{:?}", item)
    }
}
