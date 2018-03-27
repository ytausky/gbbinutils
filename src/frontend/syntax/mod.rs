mod lexer;
mod parser;

pub fn parse<'a, BC>(src: &'a str, mut actions: BC)
where
    BC: BlockContext<Terminal = StrToken<'a>>,
{
    self::parser::parse_src(self::lexer::Lexer::new(src), &mut actions)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Keyword {
    A,
    And,
    B,
    Bc,
    C,
    Cp,
    D,
    Dec,
    E,
    Endm,
    H,
    Halt,
    Hl,
    Include,
    Jr,
    L,
    Ld,
    Macro,
    Nop,
    Nz,
    Push,
    Stop,
    Xor,
    Z,
}

pub trait Token {
    fn kind(&self) -> TokenKind;
}

pub enum TokenKind {
    ClosingBracket,
    Colon,
    Comma,
    Eol,
    Identifier,
    Keyword(Keyword),
    Label,
    Number,
    OpeningBracket,
    QuotedString,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StrToken<'a> {
    ClosingBracket,
    Colon,
    Comma,
    Eol,
    Identifier(&'a str),
    Keyword(Keyword),
    Label(&'a str),
    Number(isize),
    OpeningBracket,
    QuotedString(&'a str),
}

impl<'a> Token for StrToken<'a> {
    fn kind(&self) -> TokenKind {
        match *self {
            StrToken::ClosingBracket => TokenKind::ClosingBracket,
            StrToken::Colon => TokenKind::Colon,
            StrToken::Comma => TokenKind::Comma,
            StrToken::Eol => TokenKind::Eol,
            StrToken::Identifier(_) => TokenKind::Identifier,
            StrToken::Keyword(keyword) => TokenKind::Keyword(keyword),
            StrToken::Label(_) => TokenKind::Label,
            StrToken::Number(_) => TokenKind::Number,
            StrToken::OpeningBracket => TokenKind::OpeningBracket,
            StrToken::QuotedString(_) => TokenKind::QuotedString,
        }
    }
}

impl<T: Token> Terminal for T {
    fn kind(&self) -> TerminalKind {
        match self.kind() {
            TokenKind::ClosingBracket => TerminalKind::ClosingBracket,
            TokenKind::Colon => TerminalKind::Colon,
            TokenKind::Comma => TerminalKind::Comma,
            TokenKind::Eol => TerminalKind::Eol,
            TokenKind::Identifier => TerminalKind::Word,
            TokenKind::Keyword(Keyword::Endm) => TerminalKind::Endm,
            TokenKind::Keyword(Keyword::Macro) => TerminalKind::Macro,
            TokenKind::Keyword(_) => TerminalKind::Word,
            TokenKind::Label => TerminalKind::Label,
            TokenKind::Number => TerminalKind::Number,
            TokenKind::OpeningBracket => TerminalKind::OpeningBracket,
            TokenKind::QuotedString => TerminalKind::QuotedString,
        }
    }
}

pub trait Terminal {
    fn kind(&self) -> TerminalKind;
}

#[derive(Clone, Debug, PartialEq)]
pub enum TerminalKind {
    ClosingBracket,
    Colon,
    Comma,
    Endm,
    Eol,
    Label,
    Macro,
    Number,
    OpeningBracket,
    QuotedString,
    Word,
}

pub trait BlockContext {
    type Terminal: Terminal;
    type CommandContext: CommandContext<Terminal = Self::Terminal>;
    type TerminalSequenceContext: TerminalSequenceContext<Terminal = Self::Terminal>;
    fn add_label(&mut self, label: Self::Terminal);
    fn enter_command(&mut self, name: Self::Terminal) -> &mut Self::CommandContext;
    fn enter_macro_definition(
        &mut self,
        label: Self::Terminal,
    ) -> &mut Self::TerminalSequenceContext;
}

pub trait CommandContext {
    type Terminal: Terminal;
    fn add_argument(&mut self, expr: SynExpr<Self::Terminal>);
    fn exit_command(&mut self);
}

pub trait TerminalSequenceContext {
    type Terminal: Terminal;
    fn push_terminal(&mut self, terminal: Self::Terminal);
    fn exit_terminal_sequence(&mut self);
}

#[derive(Clone, Debug, PartialEq)]
pub enum SynExpr<T> {
    Atom(T),
    Deref(Box<SynExpr<T>>),
}

impl<T> From<T> for SynExpr<T> {
    fn from(atom: T) -> Self {
        SynExpr::Atom(atom)
    }
}

impl<T> SynExpr<T> {
    pub fn deref(self) -> Self {
        SynExpr::Deref(Box::new(self))
    }
}

#[cfg(test)]
mod tests {
    use super::{Keyword, StrToken, Terminal, TerminalKind};

    #[test]
    fn colon_terminal_kind() {
        assert_eq!(StrToken::Colon.kind(), TerminalKind::Colon)
    }

    #[test]
    fn comma_terminal_kind() {
        assert_eq!(StrToken::Comma.kind(), TerminalKind::Comma)
    }

    #[test]
    fn endm_terminal_kind() {
        assert_eq!(StrToken::Keyword(Keyword::Endm).kind(), TerminalKind::Endm)
    }

    #[test]
    fn eol_terminal_kind() {
        assert_eq!(StrToken::Eol.kind(), TerminalKind::Eol)
    }

    #[test]
    fn label_terminal_kind() {
        assert_eq!(StrToken::Label("label").kind(), TerminalKind::Label)
    }

    #[test]
    fn macro_terminal_kind() {
        assert_eq!(
            StrToken::Keyword(Keyword::Macro).kind(),
            TerminalKind::Macro
        )
    }

    #[test]
    fn number_terminal_kind() {
        assert_eq!(StrToken::Number(0x1234).kind(), TerminalKind::Number)
    }

    #[test]
    fn quoted_string_terminal_kind() {
        assert_eq!(
            StrToken::QuotedString("string").kind(),
            TerminalKind::QuotedString
        )
    }

    #[test]
    fn word_terminal_kind() {
        assert_eq!(
            StrToken::Identifier("identifier").kind(),
            TerminalKind::Word
        )
    }

    #[test]
    fn opening_bracket_terminal_kind() {
        assert_eq!(
            StrToken::OpeningBracket.kind(),
            TerminalKind::OpeningBracket
        )
    }

    #[test]
    fn closing_bracket_terminal_kind() {
        assert_eq!(
            StrToken::ClosingBracket.kind(),
            TerminalKind::ClosingBracket
        )
    }
}
