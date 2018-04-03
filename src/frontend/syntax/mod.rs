mod lexer;
mod parser;

pub fn parse<'a, BC>(src: &'a str, mut actions: BC)
where
    BC: BlockContext<Terminal = StrToken<'a>>,
{
    self::parser::parse_src(self::lexer::Lexer::new(src), &mut actions)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Command {
    And,
    Cp,
    Db,
    Dec,
    Halt,
    Include,
    Jp,
    Jr,
    Ld,
    Nop,
    Push,
    Stop,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Keyword {
    A,
    B,
    Bc,
    C,
    D,
    E,
    Endm,
    H,
    Hl,
    L,
    Macro,
    Nc,
    Nz,
    Z,
}

pub trait Token {
    fn kind(&self) -> TokenKind;
}

#[derive(Clone, PartialEq)]
pub enum TokenKind {
    Command(Command),
    Identifier,
    Keyword(Keyword),
    Label,
    Number,
    QuotedString,
    Simple(SimpleTokenKind),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SimpleTokenKind {
    ClosingBracket,
    Colon,
    Comma,
    Eol,
    OpeningBracket,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StrToken<'a> {
    Command(Command),
    Identifier(&'a str),
    Keyword(Keyword),
    Label(&'a str),
    Number(isize),
    QuotedString(&'a str),
    Simple(SimpleTokenKind),
}

impl<'a> Token for StrToken<'a> {
    fn kind(&self) -> TokenKind {
        match *self {
            StrToken::Command(command) => TokenKind::Command(command),
            StrToken::Identifier(_) => TokenKind::Identifier,
            StrToken::Keyword(keyword) => TokenKind::Keyword(keyword),
            StrToken::Label(_) => TokenKind::Label,
            StrToken::Number(_) => TokenKind::Number,
            StrToken::QuotedString(_) => TokenKind::QuotedString,
            StrToken::Simple(kind) => TokenKind::Simple(kind),
        }
    }
}

impl<T: Token> Terminal for T {
    fn kind(&self) -> TerminalKind {
        match self.kind() {
            TokenKind::Command(_) => TerminalKind::Command,
            TokenKind::Simple(SimpleTokenKind::ClosingBracket) => TerminalKind::ClosingBracket,
            TokenKind::Simple(SimpleTokenKind::Colon) => TerminalKind::Colon,
            TokenKind::Simple(SimpleTokenKind::Comma) => TerminalKind::Comma,
            TokenKind::Simple(SimpleTokenKind::Eol) => TerminalKind::Eol,
            TokenKind::Keyword(Keyword::Endm) => TerminalKind::Endm,
            TokenKind::Keyword(Keyword::Macro) => TerminalKind::Macro,
            TokenKind::Identifier
            | TokenKind::Keyword(_)
            | TokenKind::Number
            | TokenKind::QuotedString => TerminalKind::Word,
            TokenKind::Label => TerminalKind::Label,
            TokenKind::Simple(SimpleTokenKind::OpeningBracket) => TerminalKind::OpeningBracket,
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
    Command,
    Endm,
    Eol,
    Label,
    Macro,
    OpeningBracket,
    Word,
}

pub trait BlockContext {
    type Terminal: Terminal;
    type CommandContext: CommandContext<Terminal = Self::Terminal>;
    type MacroInvocationContext: MacroInvocationContext<Terminal = Self::Terminal>;
    type TerminalSequenceContext: TerminalSequenceContext<Terminal = Self::Terminal>;
    fn add_label(&mut self, label: Self::Terminal);
    fn enter_command(&mut self, name: Self::Terminal) -> &mut Self::CommandContext;
    fn enter_macro_definition(
        &mut self,
        label: Self::Terminal,
    ) -> &mut Self::TerminalSequenceContext;
    fn enter_macro_invocation(&mut self, name: Self::Terminal)
        -> &mut Self::MacroInvocationContext;
}

pub trait CommandContext {
    type Terminal: Terminal;
    fn add_argument(&mut self, expr: SynExpr<Self::Terminal>);
    fn exit_command(&mut self);
}

pub trait MacroInvocationContext {
    type Terminal: Terminal;
    type TerminalSequenceContext: TerminalSequenceContext<Terminal = Self::Terminal>;
    fn enter_macro_argument(&mut self) -> &mut Self::TerminalSequenceContext;
    fn exit_macro_invocation(&mut self);
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
    use super::{Command, Keyword, SimpleTokenKind, StrToken, Terminal, TerminalKind};

    #[test]
    fn colon_terminal_kind() {
        assert_eq!(
            StrToken::Simple(SimpleTokenKind::Colon).kind(),
            TerminalKind::Colon
        )
    }

    #[test]
    fn comma_terminal_kind() {
        assert_eq!(
            StrToken::Simple(SimpleTokenKind::Comma).kind(),
            TerminalKind::Comma
        )
    }

    #[test]
    fn endm_terminal_kind() {
        assert_eq!(StrToken::Keyword(Keyword::Endm).kind(), TerminalKind::Endm)
    }

    #[test]
    fn eol_terminal_kind() {
        assert_eq!(
            StrToken::Simple(SimpleTokenKind::Eol).kind(),
            TerminalKind::Eol
        )
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
    fn nop_terminal_kind() {
        assert_eq!(
            StrToken::Command(Command::Nop).kind(),
            TerminalKind::Command,
        )
    }

    #[test]
    fn number_terminal_kind() {
        assert_eq!(StrToken::Number(0x1234).kind(), TerminalKind::Word)
    }

    #[test]
    fn quoted_string_terminal_kind() {
        assert_eq!(StrToken::QuotedString("string").kind(), TerminalKind::Word)
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
            StrToken::Simple(SimpleTokenKind::OpeningBracket).kind(),
            TerminalKind::OpeningBracket
        )
    }

    #[test]
    fn closing_bracket_terminal_kind() {
        assert_eq!(
            StrToken::Simple(SimpleTokenKind::ClosingBracket).kind(),
            TerminalKind::ClosingBracket
        )
    }
}
