//! An assembler for Game Boy.
//!
//! `gbas` is an assembler targeting Game Boy, Game Boy Pocket, Game Boy Light, and Game Boy Color.
//! Its customizable IO functions make it suitable for embedding in other tools, in unit tests, etc.

#![allow(clippy::type_complexity)]

pub use crate::codebase::FileSystem;
pub use crate::link::{Program, Rom};

use crate::diagnostics::*;

pub mod assembler;
pub mod diagnostics;

mod codebase;
mod eval;
mod expr;
mod link;
mod object;
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
