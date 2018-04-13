pub trait DiagnosticsListener {
    fn emit_diagnostic(&self, diagnostic: Diagnostic);
}

#[derive(Debug, PartialEq)]
pub enum Diagnostic {
    OperandCount { actual: usize, expected: usize },
    UndefinedMacro { name: String },
    ValueOutOfRange { value: i32, width: Width },
}

#[derive(Debug, PartialEq)]
pub enum Width {
    Byte,
}
