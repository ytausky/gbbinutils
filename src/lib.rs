mod backend;
mod diagnostics;
mod frontend;
mod ir;

pub fn analyze_file(name: &str) {
    frontend::analyze_file(name.to_string(), ir::OutputDumper::new())
}

pub fn assemble_rom(name: &str) {
    let object = backend::Rom::new();
    frontend::analyze_file(name.to_string(), object);
}
