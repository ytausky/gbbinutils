use crate::backend::Width;
use crate::codebase::{CodebaseError, TextCache};
use crate::diagnostics::span::SpanData;
use crate::instruction::IncDec;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Message<S> {
    AfOutsideStackOperation,
    AlwaysUnconditional,
    CannotBeUsedAsTarget,
    CannotDereference {
        category: KeywordOperandCategory,
        operand: S,
    },
    CannotSpecifyTarget,
    ConditionOutsideBranch,
    DestCannotBeConst,
    DestMustBeA,
    DestMustBeHl,
    ExpectedString,
    IncompatibleOperand,
    InvalidUtf8,
    IoError {
        string: String,
    },
    KeywordInExpr {
        keyword: S,
    },
    LdDerefHlDerefHl {
        mnemonic: S,
        dest: S,
        src: S,
    },
    LdSpHlOperands,
    LdWidthMismatch {
        src_width: Width,
        src: S,
        dest: S,
    },
    MacroRequiresName,
    MissingTarget,
    MustBeBit {
        mnemonic: S,
    },
    MustBeConst,
    MustBeDeref {
        operand: S,
    },
    OnlySupportedByA,
    OperandCannotBeIncDec(IncDec),
    OperandCount {
        actual: usize,
        expected: usize,
    },
    RequiresConstantTarget {
        mnemonic: S,
    },
    RequiresRegPair,
    RequiresSimpleOperand,
    SrcMustBeSp,
    StringInInstruction,
    UndefinedMacro {
        name: String,
    },
    UnexpectedEof,
    UnexpectedToken {
        token: S,
    },
    UnmatchedParenthesis,
    UnresolvedSymbol {
        symbol: String,
    },
    ValueOutOfRange {
        value: i32,
        width: Width,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum KeywordOperandCategory {
    Reg,
    RegPair,
    ConditionCode,
}

impl<S> From<CodebaseError> for Message<S> {
    fn from(error: CodebaseError) -> Message<S> {
        match error {
            CodebaseError::IoError(error) => Message::IoError {
                string: error.to_string(),
            },
            CodebaseError::Utf8Error => Message::InvalidUtf8,
        }
    }
}

impl Message<SpanData> {
    pub fn render<'a>(&self, codebase: &'a TextCache) -> String {
        use self::Message::*;
        match self {
            AfOutsideStackOperation => {
                "register pair `af` can only be used with `push` and `pop`".into()
            }
            AlwaysUnconditional => "instruction cannot be made conditional".into(),
            CannotBeUsedAsTarget => {
                "operand cannot be used as target for branching instructions".into()
            }
            CannotDereference { category, operand } => format!(
                "{} `{}` cannot be dereferenced",
                category,
                mk_snippet(codebase, operand),
            ),
            CannotSpecifyTarget => "branch target cannot be specified explicitly".into(),
            ConditionOutsideBranch => {
                "condition codes can only be used as operands for branching instructions".into()
            }
            DestCannotBeConst => "destination operand cannot be a constant".into(),
            DestMustBeA => "destination of ALU operation must be `a`".into(),
            DestMustBeHl => "destination operand must be `hl`".into(),
            ExpectedString => "expected string argument".into(),
            IncompatibleOperand => "operand cannot be used with this instruction".into(),
            InvalidUtf8 => "file contains invalid UTF-8".into(),
            IoError { string } => string.clone(),
            KeywordInExpr { keyword } => format!(
                "keyword `{}` cannot appear in expression",
                mk_snippet(codebase, keyword),
            ),
            LdDerefHlDerefHl {
                mnemonic,
                dest,
                src,
            } => format!(
                "`{} {}, {}` is not a legal instruction",
                mk_snippet(codebase, mnemonic),
                mk_snippet(codebase, dest),
                mk_snippet(codebase, src)
            ),
            LdSpHlOperands => {
                "the only legal 16-bit register to register transfer is from `hl` to `sp`".into()
            }
            LdWidthMismatch {
                src_width,
                src,
                dest,
            } => {
                let (src_bits, dest_bits) = match src_width {
                    Width::Byte => (8, 16),
                    Width::Word => (16, 8),
                };
                format!(
                    "cannot load {}-bit source `{}` into {}-bit destination `{}`",
                    src_bits,
                    mk_snippet(codebase, src),
                    dest_bits,
                    mk_snippet(codebase, dest),
                )
            }
            MacroRequiresName => "macro definition must be preceded by label".into(),
            MissingTarget => "branch instruction requires target".into(),
            MustBeBit { mnemonic } => format!(
                "first operand of `{}` must be bit number",
                mk_snippet(codebase, mnemonic),
            ),
            MustBeConst => "operand must be a constant".into(),
            MustBeDeref { operand } => format!(
                "operand `{}` must be dereferenced",
                mk_snippet(codebase, operand),
            ),
            OnlySupportedByA => "only `a` can be used for this operand".into(),
            OperandCannotBeIncDec(operation) => format!(
                "operand cannot be {}",
                match operation {
                    IncDec::Inc => "incremented",
                    IncDec::Dec => "decremented",
                }
            ),
            OperandCount { actual, expected } => format!(
                "expected {} operand{}, found {}",
                expected,
                pluralize(*expected),
                actual
            ),
            RequiresConstantTarget { mnemonic } => format!(
                "instruction `{}` requires a constant target",
                mk_snippet(codebase, mnemonic),
            ),
            RequiresRegPair => "instruction requires a register pair".into(),
            RequiresSimpleOperand => "instruction requires 8-bit register or `(hl)`".into(),
            SrcMustBeSp => "source operand must be `sp`".into(),
            StringInInstruction => "strings cannot appear in instruction operands".into(),
            UndefinedMacro { name } => format!("invocation of undefined macro `{}`", name),
            UnexpectedEof => "unexpected end of file".into(),
            UnexpectedToken { token } => format!(
                "encountered unexpected token `{}`",
                mk_snippet(codebase, token),
            ),
            UnmatchedParenthesis => "unmatched parenthesis".into(),
            UnresolvedSymbol { symbol } => format!("symbol `{}` could not be resolved", symbol),
            ValueOutOfRange { value, width } => {
                format!("value {} cannot be represented in a {}", value, width)
            }
        }
    }
}

impl fmt::Display for KeywordOperandCategory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            KeywordOperandCategory::Reg => f.write_str("register"),
            KeywordOperandCategory::RegPair => f.write_str("register pair"),
            KeywordOperandCategory::ConditionCode => f.write_str("condition code"),
        }
    }
}

fn pluralize(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

impl fmt::Display for Width {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Width::Byte => f.write_str("byte"),
            Width::Word => f.write_str("word"),
        }
    }
}

fn mk_snippet<'a>(codebase: &'a TextCache, span: &SpanData) -> &'a str {
    match span {
        SpanData::Buf { range, context } => &codebase.buf(context.buf_id).as_str()[range.clone()],
        SpanData::Macro { .. } => unimplemented!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebase::BufRange;
    use crate::diagnostics::span::BufContextData;
    use std::rc::Rc;

    #[test]
    fn get_snippet() {
        let mut codebase = TextCache::new();
        let src = "add snippet, my";
        let buf_id = codebase.add_src_buf("some_file", src);
        let context = Rc::new(BufContextData {
            buf_id,
            included_from: None,
        });
        let span = SpanData::Buf {
            range: BufRange::from(4..11),
            context: context.clone(),
        };
        assert_eq!(mk_snippet(&codebase, &span), "snippet")
    }
}
