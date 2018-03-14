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

pub trait ParsingContext {
    type Token: Terminal;

    fn enter_instruction(&mut self, name: Self::Token);
    fn exit_instruction(&mut self);

    fn enter_expression(&mut self);
    fn push_atom(&mut self, atom: Self::Token);
    fn exit_expression(&mut self);

    fn enter_macro_definition(&mut self, label: Self::Token);
    fn exit_macro_definition(&mut self);
}
