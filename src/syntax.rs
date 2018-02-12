pub trait SyntacticToken {
    fn kind(&self) -> TokenKind;
}

pub enum TokenKind {
    Comma,
    Eol,
    Number,
    QuotedString,
    Word,
}
