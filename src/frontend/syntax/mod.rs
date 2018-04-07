pub mod keyword;
pub mod lexer;
mod parser;

pub fn tokenize(src: &str) -> self::lexer::Lexer {
    self::lexer::Lexer::new(src)
}

pub fn parse_token_seq<I, BC>(tokens: I, actions: BC)
where
    I: Iterator<Item = BC::Terminal>,
    BC: BlockContext,
{
    self::parser::parse_src(tokens, actions)
}

#[derive(Clone, Debug, PartialEq)]
pub enum Token<S> {
    Atom(Atom<S>),
    ClosingBracket,
    Colon,
    Comma,
    Command(keyword::Command),
    Endm,
    Eol,
    Label(S),
    Macro,
    OpeningBracket,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Atom<S> {
    Ident(S),
    Operand(keyword::Operand),
    Number(isize),
    String(S),
}

impl<S> Terminal for Token<S> {
    fn kind(&self) -> TerminalKind {
        match *self {
            Token::Atom(_) => TerminalKind::Atom,
            Token::ClosingBracket => TerminalKind::ClosingBracket,
            Token::Colon => TerminalKind::Colon,
            Token::Comma => TerminalKind::Comma,
            Token::Command(_) => TerminalKind::Command,
            Token::Endm => TerminalKind::Endm,
            Token::Eol => TerminalKind::Eol,
            Token::Label(_) => TerminalKind::Label,
            Token::Macro => TerminalKind::Macro,
            Token::OpeningBracket => TerminalKind::OpeningBracket,
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

pub trait BlockContext
where
    Self: Sized,
{
    type Terminal: Terminal;
    type CommandContext: CommandContext<Terminal = Self::Terminal, EnclosingContext = Self>;
    type MacroDefContext: TerminalSeqContext<Terminal = Self::Terminal, EnclosingContext = Self>;
    type MacroInvocationContext: MacroInvocationContext<
        Terminal = Self::Terminal,
        EnclosingContext = Self,
    >;
    fn add_label(&mut self, label: Self::Terminal);
    fn enter_command(self, name: Self::Terminal) -> Self::CommandContext;
    fn enter_macro_def(self, name: Self::Terminal) -> Self::MacroDefContext;
    fn enter_macro_invocation(self, name: Self::Terminal) -> Self::MacroInvocationContext;
}

pub trait CommandContext {
    type Terminal: Terminal;
    type EnclosingContext;
    fn add_argument(&mut self, expr: SynExpr<Self::Terminal>);
    fn exit_command(self) -> Self::EnclosingContext;
}

pub trait MacroInvocationContext
where
    Self: Sized,
{
    type Terminal: Terminal;
    type EnclosingContext;
    type MacroArgContext: TerminalSeqContext<Terminal = Self::Terminal, EnclosingContext = Self>;
    fn enter_macro_arg(self) -> Self::MacroArgContext;
    fn exit_macro_invocation(self) -> Self::EnclosingContext;
}

pub trait TerminalSeqContext {
    type Terminal: Terminal;
    type EnclosingContext;
    fn push_terminal(&mut self, terminal: Self::Terminal);
    fn exit_terminal_seq(self) -> Self::EnclosingContext;
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
    use super::{Atom, Terminal, TerminalKind, Token, keyword::Command};

    #[test]
    fn colon_terminal_kind() {
        assert_eq!(Token::Colon::<()>.kind(), TerminalKind::Colon)
    }

    #[test]
    fn comma_terminal_kind() {
        assert_eq!(Token::Comma::<()>.kind(), TerminalKind::Comma)
    }

    #[test]
    fn endm_terminal_kind() {
        assert_eq!(Token::Endm::<()>.kind(), TerminalKind::Endm)
    }

    #[test]
    fn eol_terminal_kind() {
        assert_eq!(Token::Eol::<()>.kind(), TerminalKind::Eol)
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
            Token::OpeningBracket::<()>.kind(),
            TerminalKind::OpeningBracket
        )
    }

    #[test]
    fn closing_bracket_terminal_kind() {
        assert_eq!(
            Token::ClosingBracket::<()>.kind(),
            TerminalKind::ClosingBracket
        )
    }
}
