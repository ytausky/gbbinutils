#![no_main]
#[macro_use] extern crate libfuzzer_sys;
extern crate tempfile;

use std::io::Write;
use tempfile::NamedTempFile;

fuzz_target!(|data: &[u8]| {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(data).unwrap();
    gbas::assemble(file.path().as_os_str().to_str().unwrap(), &mut gbas::Config::default());
});
