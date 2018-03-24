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
