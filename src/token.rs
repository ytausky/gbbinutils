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
            Token::Colon => TerminalKind::Colon,
            Token::Comma => TerminalKind::Comma,
            Token::Eol => TerminalKind::Eol,
            Token::Identifier(_) => TerminalKind::Word,
            Token::Keyword(keyword::Keyword::Endm) => TerminalKind::Endm,
            Token::Keyword(keyword::Keyword::Macro) => TerminalKind::Macro,
            Token::Keyword(_) => TerminalKind::Word,
            Token::Label(_) => TerminalKind::Label,
            Token::Number(_) => TerminalKind::Number,
            Token::QuotedString(_) => TerminalKind::QuotedString,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use keyword::Keyword;
    use syntax::Terminal;
    use syntax::TerminalKind;

    #[test]
    fn colon_terminal_kind() {
        assert_eq!(Token::Colon.kind(), TerminalKind::Colon)
    }

    #[test]
    fn comma_terminal_kind() {
        assert_eq!(Token::Comma.kind(), TerminalKind::Comma)
    }

    #[test]
    fn endm_terminal_kind() {
        assert_eq!(Token::Keyword(Keyword::Endm).kind(), TerminalKind::Endm)
    }

    #[test]
    fn eol_terminal_kind() {
        assert_eq!(Token::Eol.kind(), TerminalKind::Eol)
    }

    #[test]
    fn label_terminal_kind() {
        assert_eq!(Token::Label("label").kind(), TerminalKind::Label)
    }

    #[test]
    fn macro_terminal_kind() {
        assert_eq!(Token::Keyword(Keyword::Macro).kind(), TerminalKind::Macro)
    }

    #[test]
    fn number_terminal_kind() {
        assert_eq!(Token::Number(0x1234).kind(), TerminalKind::Number)
    }

    #[test]
    fn quoted_string_terminal_kind() {
        assert_eq!(Token::QuotedString("string").kind(), TerminalKind::QuotedString)
    }

    #[test]
    fn word_terminal_kind() {
        assert_eq!(Token::Identifier("identifier").kind(), TerminalKind::Word)
    }
}
