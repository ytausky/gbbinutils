use keyword;
use syntax;

#[derive(Clone, Debug, PartialEq)]
pub enum Token<'a> {
    Comma,
    Eol,
    Keyword(keyword::Keyword),
    Number(isize),
    QuotedString(&'a str),
    Word(&'a str),
}

impl<'a> syntax::Terminal for Token<'a> {
    fn kind(&self) -> syntax::TerminalKind {
        use syntax::TerminalKind;
        match *self {
            Token::Comma => TerminalKind::Comma,
            Token::Eol => TerminalKind::Eol,
            Token::Keyword(_) => TerminalKind::Word,
            Token::Number(_) => TerminalKind::Number,
            Token::QuotedString(_) => TerminalKind::QuotedString,
            Token::Word(_) => TerminalKind::Word,
        }
    }
}
