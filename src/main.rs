extern crate gbas;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() <= 1 {
        println!("{}: error: no input files", args[0]);
        std::process::exit(1)
    }
    gbas::assemble_rom(&args[1]);
}
