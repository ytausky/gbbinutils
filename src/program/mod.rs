pub use self::builder::ProgramBuilder;

use self::context::{EvalContext, SymbolTable};
use self::resolve::Value;
use crate::diag::BackendDiagnostics;
use crate::model::Width;
use std::borrow::Borrow;

mod builder;
mod context;
mod lowering;
mod resolve;
mod translate;

type RelocExpr<S> = crate::model::RelocExpr<NameId, S>;

#[derive(Clone, Copy)]
struct ValueId(usize);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NameId(usize);

pub struct Program<S> {
    sections: Vec<Section<S>>,
    symbols: SymbolTable,
}

struct Section<S> {
    name: Option<String>,
    origin: Option<RelocExpr<S>>,
    size: ValueId,
    items: Vec<Node<S>>,
}

#[derive(Clone, Debug, PartialEq)]
enum Node<S> {
    Byte(u8),
    Expr(RelocExpr<S>, Width),
    LdInlineAddr(u8, RelocExpr<S>),
    Embedded(u8, RelocExpr<S>),
    Symbol((NameId, S), RelocExpr<S>),
}

impl<S> Program<S> {
    pub fn new() -> Program<S> {
        Program {
            sections: Vec::new(),
            symbols: SymbolTable::new(),
        }
    }

    fn add_section(&mut self, name: Option<String>) {
        let size_symbol_id = self.symbols.new_symbol(Value::Unknown);
        self.sections.push(Section::new(name, size_symbol_id))
    }
}

impl<S: Clone> Program<S> {
    pub(crate) fn link(mut self, diagnostics: &mut impl BackendDiagnostics<S>) -> BinaryObject {
        self.resolve_symbols();
        let mut context = EvalContext {
            symbols: &self.symbols,
            location: 0.into(),
        };
        BinaryObject {
            sections: self
                .sections
                .into_iter()
                .map(|section| section.translate(&mut context, diagnostics))
                .collect(),
        }
    }
}

impl<S> Section<S> {
    pub fn new(name: Option<String>, size: ValueId) -> Section<S> {
        Section {
            name,
            origin: None,
            size,
            items: Vec::new(),
        }
    }
}

impl<S: Clone> Section<S> {
    fn traverse<ST, F>(&self, context: &mut EvalContext<ST>, mut f: F) -> (Value, Value)
    where
        ST: Borrow<SymbolTable>,
        F: FnMut(&Node<S>, &mut EvalContext<ST>),
    {
        let origin = self.evaluate_origin(context);
        let mut offset = Value::from(0);
        for item in &self.items {
            offset += &item.size(&context);
            context.location = &origin + &offset;
            f(item, context)
        }
        (origin, offset)
    }

    fn evaluate_origin<ST: Borrow<SymbolTable>>(&self, context: &EvalContext<ST>) -> Value {
        self.origin
            .as_ref()
            .map(|expr| expr.evaluate(context))
            .unwrap_or_else(|| 0.into())
    }
}

pub struct BinaryObject {
    pub sections: Vec<BinarySection>,
}

impl BinaryObject {
    pub fn into_rom(self) -> Rom {
        let default = 0xffu8;
        let mut data: Vec<u8> = Vec::new();
        for section in self.sections {
            if !section.data.is_empty() {
                let end = section.origin + section.data.len();
                if data.len() < end {
                    data.resize(end, default)
                }
                data[section.origin..end].copy_from_slice(&section.data)
            }
        }
        if data.len() < MIN_ROM_LEN {
            data.resize(MIN_ROM_LEN, default)
        }
        Rom {
            data: data.into_boxed_slice(),
        }
    }
}

const MIN_ROM_LEN: usize = 0x8000;

pub struct Rom {
    pub data: Box<[u8]>,
}

pub struct BinarySection {
    pub name: Option<Box<str>>,
    pub origin: usize,
    pub data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_object_converted_to_all_0xff_rom() {
        let object = BinaryObject {
            sections: Vec::new(),
        };
        let rom = object.into_rom();
        assert_eq!(*rom.data, [0xffu8; MIN_ROM_LEN][..])
    }

    #[test]
    fn section_placed_in_rom_starting_at_origin() {
        let byte = 0x42;
        let origin = 0x150;
        let object = BinaryObject {
            sections: vec![BinarySection {
                name: None,
                origin,
                data: vec![byte],
            }],
        };
        let rom = object.into_rom();
        let mut expected = [0xffu8; MIN_ROM_LEN];
        expected[origin] = byte;
        assert_eq!(*rom.data, expected[..])
    }

    #[test]
    fn empty_section_does_not_extend_rom() {
        let origin = MIN_ROM_LEN + 1;
        let object = BinaryObject {
            sections: vec![BinarySection {
                name: None,
                origin,
                data: Vec::new(),
            }],
        };
        let rom = object.into_rom();
        assert_eq!(rom.data.len(), MIN_ROM_LEN)
    }

    #[test]
    fn new_section_has_name() {
        let name = "my_section";
        let mut program = Program::<()>::new();
        program.add_section(Some(name.into()));
        assert_eq!(program.sections[0].name, Some(name.into()))
    }
}
