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
    type ExpressionContext: ExpressionContext<Terminal = Self::Terminal>;
    fn enter_argument(&mut self) -> &mut Self::ExpressionContext;
    fn exit_command(&mut self);
}

pub trait ExpressionContext {
    type Expr;
    type Terminal: Terminal;
    fn apply_deref(&mut self);
    fn push_atom(&mut self, atom: Self::Terminal);
    fn exit_expression(&mut self);
}

pub trait TerminalSequenceContext {
    type Terminal: Terminal;
    fn push_terminal(&mut self, terminal: Self::Terminal);
    fn exit_terminal_sequence(&mut self);
}
