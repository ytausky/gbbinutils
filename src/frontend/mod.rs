use ir;

use std;

mod semantics;
mod syntax;

use ir::*;
use self::syntax::*;

pub fn analyze_file<S: ir::Section>(name: &str, section: S) {
    use std::io::prelude::*;
    let mut file = std::fs::File::open(name).unwrap();
    let mut src = String::new();
    file.read_to_string(&mut src).unwrap();
    let mut session = Session::new(section);
    let ast_builder = semantics::SemanticActions::new(&mut session);
    syntax::parse(&src, ast_builder)
}

pub trait ExprFactory {
    type Token: Token;
    fn mk_atom(&mut self, token: Self::Token) -> Expr;
}

pub struct StrExprFactory<'a>(std::marker::PhantomData<&'a ()>);

impl<'a> StrExprFactory<'a> {
    fn new() -> StrExprFactory<'a> {
        StrExprFactory(std::marker::PhantomData)
    }
}

impl<'a> ExprFactory for StrExprFactory<'a> {
    type Token = StrToken<'a>;
    fn mk_atom(&mut self, token: Self::Token) -> Expr {
        match token {
            StrToken::Identifier(ident) => Expr::Symbol(ident.to_string()),
            StrToken::Number(number) => Expr::Literal(number),
            _ => panic!(),
        }
    }
}

pub trait OperationReceiver<'a> {
    fn include_source_file(&mut self, filename: &'a str);
    fn emit_instruction(&mut self, instruction: ir::Instruction);
    fn emit_label(&mut self, label: &'a str);
}

struct Session<'a, S> {
    ast: Vec<AsmItem<'a>>,
    section: S,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AsmItem<'a> {
    Include(&'a str),
}

impl<'a, S: ir::Section> Session<'a, S> {
    fn new(section: S) -> Session<'a, S> {
        Session {
            ast: Vec::new(),
            section,
        }
    }
}

impl<'a, S: ir::Section> OperationReceiver<'a> for Session<'a, S> {
    fn include_source_file(&mut self, filename: &'a str) {
        self.ast.push(AsmItem::Include(filename))
    }

    fn emit_instruction(&mut self, instruction: ir::Instruction) {
        self.section.add_instruction(instruction)
    }

    fn emit_label(&mut self, label: &'a str) {
        self.section.add_label(label)
    }
}

#[cfg(test)]
mod tests {}
