pub trait DiagnosticsListener {
    fn emit_diagnostic(&mut self, diagnostic: Diagnostic);
}

#[derive(Debug, PartialEq)]
pub enum Diagnostic {
    OperandCount { actual: usize, expected: usize },
    UndefinedMacro { name: String },
}
