pub trait DiagnosticsListener {
    type CodeRef;
    fn emit_diagnostic(&self, diagnostic: Diagnostic<Self::CodeRef>);
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
