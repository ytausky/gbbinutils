use self::cpu_instr::mnemonic::Mnemonic;
use self::directive::Directive;

pub(super) mod cpu_instr;
pub(super) mod directive;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum BuiltinInstrMnemonic {
    Directive(Directive),
    CpuInstr(Mnemonic),
}

impl From<Directive> for BuiltinInstrMnemonic {
    fn from(directive: Directive) -> Self {
        BuiltinInstrMnemonic::Directive(directive)
    }
}

impl From<Mnemonic> for BuiltinInstrMnemonic {
    fn from(mnemonic: Mnemonic) -> Self {
        BuiltinInstrMnemonic::CpuInstr(mnemonic)
    }
}
