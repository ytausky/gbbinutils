extern crate gbas;

use std::fs::File;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() <= 1 {
        println!("{}: error: no input files", args[0]);
        std::process::exit(1)
    }
    let mut config = gbas::Config {
        input: &mut gbas::StdFileSystem::new(),
        output: &mut gbas::TerminalOutput {},
    };
    let filename = &args[1];
    if let Some(rom) = gbas::assemble(filename, &mut config) {
        let mut rom_file = File::create(filename.to_owned() + ".o").unwrap();
        rom_file.write_all(&rom.data).unwrap()
    }
}
