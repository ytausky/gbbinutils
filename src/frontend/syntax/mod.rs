mod lexer;
mod parser;

pub fn parse<'a, BC>(src: &'a str, mut actions: BC)
where
    BC: BlockContext<Terminal = Token<&'a str>>,
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
    H,
    Hl,
    L,
    Nc,
    Nz,
    Z,
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
pub enum Token<S> {
    Atom(Atom<S>),
    Command(Command),
    Endm,
    Label(S),
    Macro,
    Simple(SimpleTokenKind),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Atom<S> {
    Ident(S),
    Keyword(Keyword),
    Number(isize),
    String(S),
}

impl<S> Terminal for Token<S> {
    fn kind(&self) -> TerminalKind {
        match *self {
            Token::Command(_) => TerminalKind::Command,
            Token::Endm => TerminalKind::Endm,
            Token::Simple(SimpleTokenKind::ClosingBracket) => TerminalKind::ClosingBracket,
            Token::Simple(SimpleTokenKind::Colon) => TerminalKind::Colon,
            Token::Simple(SimpleTokenKind::Comma) => TerminalKind::Comma,
            Token::Simple(SimpleTokenKind::Eol) => TerminalKind::Eol,
            Token::Atom(_) => TerminalKind::Atom,
            Token::Label(_) => TerminalKind::Label,
            Token::Macro => TerminalKind::Macro,
            Token::Simple(SimpleTokenKind::OpeningBracket) => TerminalKind::OpeningBracket,
        }
    }
}

pub trait Terminal {
    fn kind(&self) -> TerminalKind;
}

#[derive(Clone, Debug, PartialEq)]
pub enum TerminalKind {
    Atom,
    ClosingBracket,
    Colon,
    Comma,
    Command,
    Endm,
    Eol,
    Label,
    Macro,
    OpeningBracket,
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
    use super::{Atom, Command, SimpleTokenKind, Terminal, TerminalKind, Token};

    #[test]
    fn colon_terminal_kind() {
        assert_eq!(
            Token::Simple::<()>(SimpleTokenKind::Colon).kind(),
            TerminalKind::Colon
        )
    }

    #[test]
    fn comma_terminal_kind() {
        assert_eq!(
            Token::Simple::<()>(SimpleTokenKind::Comma).kind(),
            TerminalKind::Comma
        )
    }

    #[test]
    fn endm_terminal_kind() {
        assert_eq!(Token::Endm::<()>.kind(), TerminalKind::Endm)
    }

    #[test]
    fn eol_terminal_kind() {
        assert_eq!(
            Token::Simple::<()>(SimpleTokenKind::Eol).kind(),
            TerminalKind::Eol
        )
    }

    #[test]
    fn label_terminal_kind() {
        assert_eq!(Token::Label("label").kind(), TerminalKind::Label)
    }

    #[test]
    fn macro_terminal_kind() {
        assert_eq!(Token::Macro::<()>.kind(), TerminalKind::Macro)
    }

    #[test]
    fn nop_terminal_kind() {
        assert_eq!(
            Token::Command::<()>(Command::Nop).kind(),
            TerminalKind::Command,
        )
    }

    #[test]
    fn number_terminal_kind() {
        assert_eq!(
            Token::Atom::<()>(Atom::Number(0x1234)).kind(),
            TerminalKind::Atom
        )
    }

    #[test]
    fn quoted_string_terminal_kind() {
        assert_eq!(
            Token::Atom(Atom::String("string")).kind(),
            TerminalKind::Atom
        )
    }

    #[test]
    fn word_terminal_kind() {
        assert_eq!(
            Token::Atom(Atom::Ident("identifier")).kind(),
            TerminalKind::Atom
        )
    }

    #[test]
    fn opening_bracket_terminal_kind() {
        assert_eq!(
            Token::Simple::<()>(SimpleTokenKind::OpeningBracket).kind(),
            TerminalKind::OpeningBracket
        )
    }

    #[test]
    fn closing_bracket_terminal_kind() {
        assert_eq!(
            Token::Simple::<()>(SimpleTokenKind::ClosingBracket).kind(),
            TerminalKind::ClosingBracket
        )
    }
}
