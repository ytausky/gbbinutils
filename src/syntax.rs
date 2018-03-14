pub trait Terminal {
    fn kind(&self) -> TerminalKind;
}

#[derive(Clone, Debug, PartialEq)]
pub enum TerminalKind {
    Colon,
    Comma,
    Endm,
    Eol,
    Label,
    Macro,
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

pub trait ParsingContext {
    type Token: Terminal;
    type Item;
    type Expr: Expr<Terminal = Self::Token>;
    type Block: Block<Item = Self::Item>;

    fn enter_instruction(&mut self, name: Self::Token);
    fn exit_instruction(&mut self);

    fn enter_expression(&mut self);
    fn push_identifier(&mut self, identifier: Self::Token);
    fn push_literal(&mut self, literal: Self::Token);
    fn exit_expression(&mut self);

    fn enter_macro_definition(&mut self, label: Self::Token);
    fn exit_macro_definition(&mut self);

    fn define_macro(&mut self, label: Self::Token, block: Self::Block) -> Self::Item;
    fn reduce_command(&mut self, name: Self::Token, args: &[Self::Expr]) -> Self::Item;
}
