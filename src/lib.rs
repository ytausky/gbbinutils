//! An assembler for Game Boy.
//!
//! `gbas` is an assembler targeting Game Boy, Game Boy Pocket, Game Boy Light, and Game Boy Color.
//! Its customizable IO functions make it suitable for embedding in other tools, in unit tests, etc.

#![allow(clippy::type_complexity)]

pub use crate::assembler::Assembler;
pub use crate::codebase::FileSystem;
pub use crate::diagnostics::*;
pub use crate::linker::Linker;
pub use crate::object::Object;
pub use crate::program::{Chunk, Program};

mod assembler;
mod codebase;
mod diagnostics;
mod eval;
mod expr;
mod linker;
mod object;
mod program;
mod span;

#[cfg(test)]
mod log;

#[derive(Default)]
pub struct Config<'a> {
    pub input: InputConfig<'a>,
    pub diagnostics: DiagnosticsConfig<'a>,
}

pub enum InputConfig<'a> {
    Default,
    Custom(&'a mut dyn codebase::FileSystem),
}

impl<'a> Default for InputConfig<'a> {
    fn default() -> Self {
        InputConfig::Default
    }
}

pub enum DiagnosticsConfig<'a> {
    Ignore,
    Output(&'a mut dyn FnMut(Diagnostic)),
}

impl<'a> Default for DiagnosticsConfig<'a> {
    fn default() -> Self {
        DiagnosticsConfig::Ignore
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum IncDec {
    Inc,
    Dec,
}
