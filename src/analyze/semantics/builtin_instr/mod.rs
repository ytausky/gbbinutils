use self::cpu_instr::mnemonic::Mnemonic;
use self::directive::Directive;

pub(super) mod cpu_instr;
pub(super) mod directive;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum BuiltinInstr {
    Directive(Directive),
    Mnemonic(Mnemonic),
}

impl From<Directive> for BuiltinInstr {
    fn from(directive: Directive) -> Self {
        BuiltinInstr::Directive(directive)
    }
}

impl From<Mnemonic> for BuiltinInstr {
    fn from(mnemonic: Mnemonic) -> Self {
        BuiltinInstr::Mnemonic(mnemonic)
    }
}
