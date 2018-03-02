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

pub trait Block {
    type Item;
    
    fn new() -> Self;
    fn push(&mut self, item: Self::Item);
}

pub trait Expr {
    type Terminal: Terminal;

    fn from_terminal(terminal: Self::Terminal) -> Self;
}

pub trait ProductionRules {
    type Token: Terminal;
    type Item;
    type Expr: Expr<Terminal = Self::Token>;
    type Block: Block<Item = Self::Item>;

    fn reduce_command(&mut self, name: Self::Token, args: &[Self::Expr]) -> Self::Item;
}
