use keyword;
use syntax;

#[derive(Clone, Debug, PartialEq)]
pub enum Token<'a> {
    Colon,
    Comma,
    Eol,
    Identifier(&'a str),
    Keyword(keyword::Keyword),
    Label(&'a str),
    Number(isize),
    QuotedString(&'a str),
}

impl<'a> syntax::Terminal for Token<'a> {
    fn kind(&self) -> syntax::TerminalKind {
        use syntax::TerminalKind;
        match *self {
            Token::Comma => TerminalKind::Comma,
            Token::Eol => TerminalKind::Eol,
            Token::Identifier(_) => TerminalKind::Word,
            Token::Keyword(_) => TerminalKind::Word,
            Token::Label(_) => TerminalKind::Label,
            Token::Number(_) => TerminalKind::Number,
            Token::QuotedString(_) => TerminalKind::QuotedString,
            _ => panic!(),
        }
    }
}
