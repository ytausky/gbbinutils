use backend::link::{UndefinedSymbol, Value};
use backend::Data;
use diagnostics::{Diagnostic, Message, Source, SourceInterval};
use instruction::RelocExpr;
use std::borrow::Borrow;
use std::collections::HashMap;
use Width;

pub struct LinkingContext {
    symbols: Vec<Option<Value>>,
    names: HashMap<String, SymbolId>,
    sizes: Vec<SymbolId>,
}

#[derive(Clone, Copy)]
pub struct SymbolId(usize);

pub trait SymbolRef {
    fn associate(&self, context: &mut LinkingContext, id: SymbolId);
    fn to_symbol_id(&self, table: &LinkingContext) -> Option<SymbolId>;
}

impl SymbolRef for SymbolId {
    fn associate(&self, _: &mut LinkingContext, _: SymbolId) {}

    fn to_symbol_id(&self, _: &LinkingContext) -> Option<SymbolId> {
        Some((*self).clone())
    }
}

impl<Q: Borrow<str>> SymbolRef for Q {
    fn associate(&self, context: &mut LinkingContext, id: SymbolId) {
        context.names.insert(self.borrow().to_string(), id);
    }

    fn to_symbol_id(&self, table: &LinkingContext) -> Option<SymbolId> {
        table.names.get(self.borrow()).cloned()
    }
}

pub struct ChunkSize(pub usize);

impl SymbolRef for ChunkSize {
    fn associate(&self, context: &mut LinkingContext, id: SymbolId) {
        let ChunkSize(index) = *self;
        assert_eq!(index, context.sizes.len());
        context.sizes.push(id)
    }

    fn to_symbol_id(&self, context: &LinkingContext) -> Option<SymbolId> {
        let ChunkSize(index) = *self;
        context.sizes.get(index).cloned()
    }
}

impl LinkingContext {
    pub fn new() -> LinkingContext {
        LinkingContext {
            symbols: Vec::new(),
            names: HashMap::new(),
            sizes: Vec::new(),
        }
    }

    pub fn define(&mut self, key: impl SymbolRef, value: Option<Value>) {
        let id = SymbolId(self.symbols.len());
        self.symbols.push(value);
        key.associate(self, id)
    }

    pub fn get(&self, key: impl SymbolRef) -> Option<&Option<Value>> {
        key.to_symbol_id(self).map(|SymbolId(id)| &self.symbols[id])
    }

    fn get_mut(&mut self, key: impl SymbolRef) -> Option<&mut Option<Value>> {
        key.to_symbol_id(self)
            .map(move |SymbolId(id)| &mut self.symbols[id])
    }

    pub fn refine(&mut self, key: impl SymbolRef, value: Value) -> bool {
        let stored_value = self.get_mut(key).unwrap();
        let old_value = stored_value.clone();
        let was_refined = old_value.map_or(true, |v| value.len() < v.len());
        *stored_value = Some(value);
        was_refined
    }

    pub fn resolve_expr_item<SR: SourceInterval>(
        &self,
        expr: &RelocExpr<SR>,
        width: Width,
    ) -> Result<Data, Diagnostic<SR>> {
        let range = expr.source_interval();
        let value = expr.evaluate(self)
            .map_err(|undefined| {
                let UndefinedSymbol(symbol, range) = undefined;
                Diagnostic::new(Message::UnresolvedSymbol { symbol }, range)
            })?
            .unwrap()
            .exact()
            .unwrap();
        fit_to_width((value, range), width)
    }
}

fn fit_to_width<SR: Clone>(
    (value, value_ref): (i32, SR),
    width: Width,
) -> Result<Data, Diagnostic<SR>> {
    if !is_in_range(value, width) {
        Err(Diagnostic::new(
            Message::ValueOutOfRange { value, width },
            value_ref.clone(),
        ))
    } else {
        Ok(match width {
            Width::Byte => Data::Byte(value as u8),
            Width::Word => Data::Word(value as u16),
        })
    }
}

fn is_in_range(n: i32, width: Width) -> bool {
    match width {
        Width::Byte => is_in_byte_range(n),
        Width::Word => true,
    }
}

fn is_in_byte_range(n: i32) -> bool {
    is_in_i8_range(n) || is_in_u8_range(n)
}

fn is_in_i8_range(n: i32) -> bool {
    n >= i32::from(i8::min_value()) && n <= i32::from(i8::max_value())
}

fn is_in_u8_range(n: i32) -> bool {
    n >= i32::from(u8::min_value()) && n <= i32::from(u8::max_value())
}
