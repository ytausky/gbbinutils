pub trait Terminal {
    fn kind(&self) -> TerminalKind;
}

#[derive(Clone, Debug, PartialEq)]
pub enum TerminalKind {
    Colon,
    Comma,
    Endm,
    Eol,
    Label,
    Macro,
    Number,
    QuotedString,
    Word,
}

pub trait BlockContext {
    type Terminal: Terminal;
    type InstructionContext: InstructionContext<Terminal = Self::Terminal>;
    fn enter_instruction(&mut self, name: Self::Terminal) -> &mut Self::InstructionContext;
    fn enter_macro_definition(&mut self, label: Self::Terminal);
    fn exit_block(&mut self);
}

pub trait InstructionContext {
    type Terminal: Terminal;
    type ExpressionContext: ExpressionContext<Terminal = Self::Terminal>;
    fn enter_argument(&mut self) -> &mut Self::ExpressionContext;
    fn exit_instruction(&mut self);
}

pub trait ExpressionContext {
    type Terminal: Terminal;
    fn push_atom(&mut self, atom: Self::Terminal);
    fn exit_expression(&mut self);
}
