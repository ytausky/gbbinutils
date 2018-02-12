pub trait Terminal {
    fn kind(&self) -> TerminalKind;
}

pub enum TerminalKind {
    Comma,
    Eol,
    Number,
    QuotedString,
    Word,
}
