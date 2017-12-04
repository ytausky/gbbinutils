#[derive(Debug, PartialEq)]
pub struct EmitBytes {
    pub mnemonic: String,
    pub operands: Vec<Operand>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operand {
    #[cfg(test)]
    RegisterPair(RegisterPair),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegisterPair {
    #[cfg(test)]
    Bc,
}

#[cfg(test)]
pub type AssemblyCommands = Vec<EmitBytes>;
