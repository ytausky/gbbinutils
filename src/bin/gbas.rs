use gbbinutils::*;

use std::fs::File;
use std::io::Write;

fn main() {
    let mut args = std::env::args();
    args.next();
    let args: Vec<String> = args.collect();
    if args.len() <= 0 {
        println!("{}: error: no input files", args[0]);
        std::process::exit(1)
    }
    let mut diagnostics = |diagnostic| println!("{}", diagnostic);
    let mut config = Config {
        input: InputConfig::default(),
        diagnostics: DiagnosticsConfig::Output(&mut diagnostics),
    };
    let mut assembler = Assembler::new(&mut config);
    let objects: Vec<_> = args
        .iter()
        .map(|path| {
            assembler
                .assemble(path)
                .unwrap_or_else(|| std::process::exit(1))
        })
        .collect();
    let mut linker = Linker::new(&mut config);
    let program = linker.link(objects).unwrap();
    let mut rom_file = File::create(args[0].to_owned() + ".o").unwrap();
    rom_file.write_all(&program.into_rom()).unwrap()
}
