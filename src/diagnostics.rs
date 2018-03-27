#[derive(Debug, PartialEq)]
pub enum Error {
    OperandCount(usize, usize),
}
