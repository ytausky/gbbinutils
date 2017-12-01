#[derive(Debug, PartialEq)]
pub struct EmitBytes {
    pub mnemonic: String,
}

#[cfg(test)]
pub type AssemblyCommands = Vec<EmitBytes>;
