#[derive(Debug, PartialEq)]
pub struct EmitBytes {
    pub mnemonic: String,
    pub operands: Vec<String>,
}

#[cfg(test)]
pub type AssemblyCommands = Vec<EmitBytes>;
