pub trait DiagnosticsListener {
    fn emit_diagnostic(&mut self, diagnostic: Error);
}

#[derive(Debug, PartialEq)]
pub enum Error {
    OperandCount { actual: usize, expected: usize },
    UndefinedMacro { name: String },
}
