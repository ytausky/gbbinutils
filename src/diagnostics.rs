pub trait DiagnosticsListener<R> {
    fn emit_diagnostic(&self, diagnostic: Diagnostic<R>);
}

#[derive(Debug, PartialEq)]
pub enum Diagnostic<R> {
    OperandCount { actual: usize, expected: usize },
    UndefinedMacro { name: (String, R) },
    ValueOutOfRange { value: i32, width: Width },
}

#[derive(Debug, PartialEq)]
pub enum Width {
    Byte,
}
