#[derive(Debug, PartialEq)]
pub struct EmitBytes {
    pub mnemonic: String,
    pub operands: Vec<Operand>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operand {
    #[cfg(test)]
    Register(Register),
    #[cfg(test)]
    RegisterPair(RegisterPair),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Register {
    #[cfg(test)]
    A,
    #[cfg(test)]
    B,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegisterPair {
    #[cfg(test)]
    Bc,
}

#[cfg(test)]
pub type AssemblyCommands = Vec<EmitBytes>;

#[cfg(test)]
pub fn make_emit_bytes(mnemonic: &str, operands: &[Operand]) -> EmitBytes {
    EmitBytes {
        mnemonic: mnemonic.to_owned(),
        operands: operands.iter().cloned().collect(),
    }
}
