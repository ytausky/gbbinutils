use syntax;

#[derive(Clone, Debug, PartialEq)]
pub enum Token<'a> {
    Comma,
    Eol,
    Number(isize),
    QuotedString(&'a str),
    Word(&'a str),
}

impl<'a> syntax::SyntacticToken for Token<'a> {
    fn kind(&self) -> syntax::TokenKind {
        use syntax::TokenKind;
        match *self {
            Token::Comma => TokenKind::Comma,
            Token::Eol => TokenKind::Eol,
            Token::Number(_) => TokenKind::Number,
            Token::QuotedString(_) => TokenKind::QuotedString,
            Token::Word(_) => TokenKind::Word,
        }
    }
}
