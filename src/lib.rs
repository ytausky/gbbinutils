mod ast;
mod codegen;
mod keyword;
mod lexer;
mod parse;
mod semantics;
mod syntax;
mod token;

pub struct AnalyzedSrc;

pub fn analyze_file(name: &str) -> AnalyzedSrc {
    use std::io::prelude::*;
    let mut file = std::fs::File::open(name).unwrap();
    let mut src = String::new();
    file.read_to_string(&mut src).unwrap();
    let ast = parse::parse_src(lexer::Lexer::new(&src), &mut semantics::AstBuilder::new());
    dump_ast(&ast);
    AnalyzedSrc {}
}

fn dump_ast(ast: &Vec<ast::AsmItem>) {
    for ref item in ast {
        println!("{:?}", item)
    }
}

#[cfg(test)]
struct FirstPass;

#[cfg(test)]
impl FirstPass {
    #[cfg(test)]
    fn new(_src: &str) -> FirstPass {
        FirstPass {}
    }

    #[cfg(test)]
    fn sections(&self) -> SectionIterator {
        SectionIterator {}
    }
}

#[cfg(test)]
type Section = u8;

#[cfg(test)]
struct SectionIterator;

#[cfg(test)]
impl Iterator for SectionIterator {
    type Item = Section;

    fn next(&mut self) -> Option<Section> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_pass_on_empty_src() {
        let first_pass = FirstPass::new("");
        assert_eq!(first_pass.sections().next(), None);
    }
}
