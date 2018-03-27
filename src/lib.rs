mod codegen;
mod diagnostics;
mod frontend;
mod ir;

pub fn analyze_file(name: &str) {
    frontend::analyze_file(name, ir::DumpSection::new())
}
