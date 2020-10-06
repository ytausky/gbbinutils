use gbbinutils::*;

use std::fs::File;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() <= 1 {
        println!("{}: error: no input files", args[0]);
        std::process::exit(1)
    }
    let filename = &args[1];
    let mut diagnostics = |diagnostic| println!("{}", diagnostic);
    let mut config = Config {
        input: InputConfig::default(),
        diagnostics: DiagnosticsConfig::Output(&mut diagnostics),
    };
    if let Some(program) = assemble(filename, &mut config) {
        let mut rom_file = File::create(filename.to_owned() + ".o").unwrap();
        rom_file.write_all(&program.into_rom().data).unwrap()
    }
}
