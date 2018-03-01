pub trait Terminal {
    fn kind(&self) -> TerminalKind;
}

#[derive(Clone, Debug, PartialEq)]
pub enum TerminalKind {
    Comma,
    Eol,
    Number,
    QuotedString,
    Word,
}

pub trait ProductionRules {
    type Token: Terminal;
    type Item;
    type Expr;

    fn build_name_expr(&mut self, token: Self::Token) -> Self::Expr;
    fn reduce_command(&mut self, name: Self::Token, args: &[Self::Expr]) -> Self::Item;
}
