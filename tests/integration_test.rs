use gbas;

use std::io;

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

fn assemble_snippet(src: &str) -> (Option<Box<[u8]>>, Vec<gbas::diag::Diagnostic>) {
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
