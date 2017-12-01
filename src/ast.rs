#[derive(Debug, PartialEq)]
pub struct EmitBytes {
    mnemonic: String,
}

pub type AssemblyCommands = Vec<EmitBytes>;
