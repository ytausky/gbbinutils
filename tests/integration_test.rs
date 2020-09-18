use std::io;

#[test]
fn assemble_empty_line_at_end() {
    assert_eq!(
        assemble_snippet(
            r"
        NOP
"
        ),
        (Some(vec![NOP].into()), vec![])
    )
}

#[test]
fn undefined_global_names_do_not_interfere_with_local_names() {
    let src = r"
GLOBAL  AND     A
        JR      Z, _SKIP
        CALL    OTHER
_SKIP   NOP

OTHER   RET";
    let binary = [
        0xa7, // AND A
        0x28, 0x03, // JR Z, _SKIP
        0xcd, 0x07, 0x00, // CALL OTHER
        0x00, // NOP
        0xc9, // RET
    ];
    let (assembled, diagnostics) = assemble_snippet(src);
    assert_eq!(diagnostics, []);
    assert_eq!(*assembled.unwrap(), binary)
}

#[test]
fn dispatch_instrs_in_taken_if() {
    assert_eq!(
        assemble_snippet(
            r"
        IF      1
        NOP
        ENDC
        NOP"
        ),
        (Some(vec![NOP, NOP].into()), vec![])
    )
}

#[test]
fn ignore_instrs_in_untaken_if() {
    assert_eq!(
        assemble_snippet(
            r"
        IF      0
        NOP
        ENDC
        NOP"
        ),
        (Some(vec![NOP].into()), vec![])
    )
}

#[test]
fn new_line_after_macro_args() {
    assert_eq!(
        assemble_snippet(
            r"
MY_MAC(OP)
        MACRO
        OP
        ENDM
        MY_MAC NOP
"
        ),
        (Some(vec![NOP].into()), vec![])
    )
}

const NOP: u8 = 0x00;

fn assemble_snippet(src: &str) -> (Option<Box<[u8]>>, Vec<gbas::diagnostics::Diagnostic>) {
    let name = "__buffer";
    let mut fs = SingleBuffer::new(name, src);
    let mut diagnostics = vec![];
    let mut output = |diagnostic| diagnostics.push(diagnostic);
    let mut config = gbas::Config {
        input: gbas::InputConfig::Custom(&mut fs),
        diagnostics: gbas::DiagnosticsConfig::Output(&mut output),
    };
    let binary = gbas::assemble(name, &mut config);
    (
        binary.map(|mut binary| binary.sections.pop().unwrap().data.into()),
        diagnostics,
    )
}

struct SingleBuffer<'a> {
    name: &'a str,
    src: &'a str,
}

impl<'a> SingleBuffer<'a> {
    fn new(name: &'a str, src: &'a str) -> Self {
        Self { name, src }
    }
}

impl<'a> gbas::FileSystem for SingleBuffer<'a> {
    fn read_file(&self, name: &str) -> Result<Vec<u8>, io::Error> {
        if name == self.name {
            Ok(self.src.bytes().collect())
        } else {
            panic!()
        }
    }
}
