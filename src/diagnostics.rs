#[derive(Debug, PartialEq)]
pub enum Error {
    OperandCount { actual: usize, expected: usize },
}
