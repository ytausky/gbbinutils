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

pub trait ParsingContext {
    type Terminal: Terminal;
    type ExpressionContext: ExpressionContext<Terminal = Self::Terminal>;

    fn enter_instruction(&mut self, name: Self::Terminal);
    fn exit_instruction(&mut self);

    fn enter_expression(&mut self) -> &mut Self::ExpressionContext;

    fn enter_macro_definition(&mut self, label: Self::Terminal);
    fn exit_macro_definition(&mut self);
}

pub trait ExpressionContext {
    type Terminal: Terminal;
    fn push_atom(&mut self, atom: Self::Terminal);
    fn exit_expression(&mut self);
}
