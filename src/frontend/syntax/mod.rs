mod lexer;
mod parser;

pub fn parse<'a, BC>(src: &'a str, mut actions: BC)
where
    BC: BlockContext<Terminal = StrToken<&'a str>>,
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
pub enum StrToken<S> {
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

impl<S> Terminal for StrToken<S> {
    fn kind(&self) -> TerminalKind {
        match *self {
            StrToken::Command(_) => TerminalKind::Command,
            StrToken::Endm => TerminalKind::Endm,
            StrToken::Simple(SimpleTokenKind::ClosingBracket) => TerminalKind::ClosingBracket,
            StrToken::Simple(SimpleTokenKind::Colon) => TerminalKind::Colon,
            StrToken::Simple(SimpleTokenKind::Comma) => TerminalKind::Comma,
            StrToken::Simple(SimpleTokenKind::Eol) => TerminalKind::Eol,
            StrToken::Atom(_) => TerminalKind::Atom,
            StrToken::Label(_) => TerminalKind::Label,
            StrToken::Macro => TerminalKind::Macro,
            StrToken::Simple(SimpleTokenKind::OpeningBracket) => TerminalKind::OpeningBracket,
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
    use super::{Atom, Command, SimpleTokenKind, StrToken, Terminal, TerminalKind};

    #[test]
    fn colon_terminal_kind() {
        assert_eq!(
            StrToken::Simple::<()>(SimpleTokenKind::Colon).kind(),
            TerminalKind::Colon
        )
    }

    #[test]
    fn comma_terminal_kind() {
        assert_eq!(
            StrToken::Simple::<()>(SimpleTokenKind::Comma).kind(),
            TerminalKind::Comma
        )
    }

    #[test]
    fn endm_terminal_kind() {
        assert_eq!(StrToken::Endm::<()>.kind(), TerminalKind::Endm)
    }

    #[test]
    fn eol_terminal_kind() {
        assert_eq!(
            StrToken::Simple::<()>(SimpleTokenKind::Eol).kind(),
            TerminalKind::Eol
        )
    }

    #[test]
    fn label_terminal_kind() {
        assert_eq!(StrToken::Label("label").kind(), TerminalKind::Label)
    }

    #[test]
    fn macro_terminal_kind() {
        assert_eq!(StrToken::Macro::<()>.kind(), TerminalKind::Macro)
    }

    #[test]
    fn nop_terminal_kind() {
        assert_eq!(
            StrToken::Command::<()>(Command::Nop).kind(),
            TerminalKind::Command,
        )
    }

    #[test]
    fn number_terminal_kind() {
        assert_eq!(
            StrToken::Atom::<()>(Atom::Number(0x1234)).kind(),
            TerminalKind::Atom
        )
    }

    #[test]
    fn quoted_string_terminal_kind() {
        assert_eq!(
            StrToken::Atom(Atom::String("string")).kind(),
            TerminalKind::Atom
        )
    }

    #[test]
    fn word_terminal_kind() {
        assert_eq!(
            StrToken::Atom(Atom::Ident("identifier")).kind(),
            TerminalKind::Atom
        )
    }

    #[test]
    fn opening_bracket_terminal_kind() {
        assert_eq!(
            StrToken::Simple::<()>(SimpleTokenKind::OpeningBracket).kind(),
            TerminalKind::OpeningBracket
        )
    }

    #[test]
    fn closing_bracket_terminal_kind() {
        assert_eq!(
            StrToken::Simple::<()>(SimpleTokenKind::ClosingBracket).kind(),
            TerminalKind::ClosingBracket
        )
    }
}
