use frontend::ast;

mod lexer;
mod parser;

pub fn parse<'a, BC>(src: &'a str, mut actions: BC)
where
    BC: BlockContext<Terminal = Token<'a>, Expr = ast::Expression<Token<'a>>>,
{
    self::parser::parse_src(
        self::lexer::Lexer::new(src),
        &mut actions,
        ast::ExprBuilder::new(),
    )
}

#[derive(Clone, Debug, PartialEq)]
pub enum Keyword {
    A,
    And,
    B,
    Bc,
    Endm,
    Halt,
    Hl,
    Include,
    Ld,
    Macro,
    Nop,
    Push,
    Stop,
    Xor,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Token<'a> {
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

impl<'a> Terminal for Token<'a> {
    fn kind(&self) -> TerminalKind {
        match *self {
            Token::ClosingBracket => TerminalKind::ClosingBracket,
            Token::Colon => TerminalKind::Colon,
            Token::Comma => TerminalKind::Comma,
            Token::Eol => TerminalKind::Eol,
            Token::Identifier(_) => TerminalKind::Word,
            Token::Keyword(Keyword::Endm) => TerminalKind::Endm,
            Token::Keyword(Keyword::Macro) => TerminalKind::Macro,
            Token::Keyword(_) => TerminalKind::Word,
            Token::Label(_) => TerminalKind::Label,
            Token::Number(_) => TerminalKind::Number,
            Token::OpeningBracket => TerminalKind::OpeningBracket,
            Token::QuotedString(_) => TerminalKind::QuotedString,
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
    type Expr;
    type CommandContext: CommandContext<Expr = Self::Expr>;
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
    type Expr;
    fn add_argument(&mut self, expr: Self::Expr);
    fn exit_command(&mut self);
}

pub trait TerminalSequenceContext {
    type Terminal: Terminal;
    fn push_terminal(&mut self, terminal: Self::Terminal);
    fn exit_terminal_sequence(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            Token::QuotedString("string").kind(),
            TerminalKind::QuotedString
        )
    }

    #[test]
    fn word_terminal_kind() {
        assert_eq!(Token::Identifier("identifier").kind(), TerminalKind::Word)
    }

    #[test]
    fn opening_bracket_terminal_kind() {
        assert_eq!(Token::OpeningBracket.kind(), TerminalKind::OpeningBracket)
    }

    #[test]
    fn closing_bracket_terminal_kind() {
        assert_eq!(Token::ClosingBracket.kind(), TerminalKind::ClosingBracket)
    }
}
