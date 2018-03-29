use ir;

use std;

mod semantics;
mod syntax;

use ir::*;
use self::syntax::*;

pub fn analyze_file<S: ir::Section>(name: &str, section: S) {
    let mut session = Session::new(section);
    session.include_source_file(name);
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

pub trait OperationReceiver<'src> {
    fn include_source_file(&mut self, filename: &'src str);
    fn emit_instruction(&mut self, instruction: ir::Instruction);
    fn emit_label(&mut self, label: &'src str);
}

struct Session<'session, S> {
    section: S,
    _phantom: std::marker::PhantomData<&'session ()>,
}

impl<'session, S: ir::Section> Session<'session, S> {
    fn new(section: S) -> Session<'session, S> {
        Session {
            section,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'src, 'session: 'src, S: ir::Section> OperationReceiver<'src> for Session<'session, S> {
    fn include_source_file(&mut self, filename: &'src str) {
        use std::io::prelude::*;
        let mut file = std::fs::File::open(filename).unwrap();
        let mut src = String::new();
        file.read_to_string(&mut src).unwrap();
        let actions = semantics::SemanticActions::new(self);
        syntax::parse(&src, actions)
    }

    fn emit_instruction(&mut self, instruction: ir::Instruction) {
        self.section.add_instruction(instruction)
    }

    fn emit_label(&mut self, label: &'src str) {
        self.section.add_label(label)
    }
}

#[cfg(test)]
mod tests {}
