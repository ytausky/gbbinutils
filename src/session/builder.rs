use super::SessionImpl;

use crate::codebase::{BufId, BufRange};
use crate::diag::span::RcSpan;
use crate::diag::Diagnostics;
use crate::expr::Expr;
use crate::object::*;
use crate::{BuiltinSymbols, CompositeSession};

pub(crate) trait Backend<S: Clone>: AllocSymbol<S> {
    fn define_symbol(&mut self, name: Self::SymbolId, span: S, expr: Expr<Self::SymbolId, S>);
    fn emit_fragment(&mut self, fragment: Fragment<Expr<Self::SymbolId, S>>);
    fn is_non_zero(&mut self, value: Expr<Self::SymbolId, S>) -> Option<bool>;
    fn set_origin(&mut self, origin: Expr<Self::SymbolId, S>);
    fn start_section(&mut self, name: Self::SymbolId, span: S);
}

pub trait AllocSymbol<S: Clone>: SymbolSource {
    fn alloc_symbol(&mut self, span: S) -> Self::SymbolId;
}

pub trait SymbolSource {
    type SymbolId: Clone;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Width {
    Byte,
    Word,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AluOperation {
    Add,
    Adc,
    Sub,
    Sbc,
    And,
    Xor,
    Or,
    Cp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BitOperation {
    Bit,
    Set,
    Res,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MiscOperation {
    Rlc,
    Rrc,
    Rl,
    Rr,
    Sla,
    Sra,
    Swap,
    Srl,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum M {
    A,
    B,
    C,
    D,
    E,
    H,
    L,
    DerefHl,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Direction {
    FromA,
    IntoA,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reg16 {
    Bc,
    De,
    Hl,
    Sp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegPair {
    Bc,
    De,
    Hl,
    Af,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PtrReg {
    Bc,
    De,
    Hli,
    Hld,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IncDec {
    Inc,
    Dec,
}

pub(crate) struct ObjectBuilder<'a, S> {
    context: LinkageContext<&'a mut Content<S>, &'a mut VarTable>,
    state: Option<BuilderState<S>>,
}

enum BuilderState<S> {
    AnonSectionPrelude { addr: Option<Expr<SymbolId, S>> },
    Section(usize),
    SectionPrelude(usize),
}

impl<'a, S> ObjectBuilder<'a, S> {
    pub fn new(Object { content, vars }: &'a mut Object<S>) -> Self {
        ObjectBuilder {
            context: LinkageContext {
                content,
                vars,
                location: 0.into(),
            },
            state: Some(BuilderState::AnonSectionPrelude { addr: None }),
        }
    }

    fn push(&mut self, fragment: Fragment<Expr<SymbolId, S>>) {
        self.current_section().fragments.push(fragment)
    }

    fn current_section(&mut self) -> &mut Section<S> {
        match self.state.take().unwrap() {
            BuilderState::AnonSectionPrelude { addr } => {
                self.add_section(None);
                let index = self.context.content.sections.len() - 1;
                self.state = Some(BuilderState::Section(index));
                let section = &mut self.context.content.sections[index];
                section.constraints.addr = addr;
                section
            }
            BuilderState::SectionPrelude(index) | BuilderState::Section(index) => {
                self.state = Some(BuilderState::Section(index));
                &mut self.context.content.sections[index]
            }
        }
    }

    fn add_section(&mut self, symbol: Option<ContentId>) {
        self.context.content.add_section(
            symbol,
            self.context.vars.alloc(),
            self.context.vars.alloc(),
        )
    }
}

impl<'a, 'b, 'c> SymbolSource for SessionImpl<'a, 'b, 'c> {
    type SymbolId = SymbolId;
}

impl<'a, 'b, 'c> AllocSymbol<RcSpan<BufId, BufRange>> for SessionImpl<'a, 'b, 'c> {
    fn alloc_symbol(&mut self, span: RcSpan<BufId, BufRange>) -> Self::SymbolId {
        self.builder.alloc_symbol(span)
    }
}

impl<'a, 'b, 'c> Backend<RcSpan<BufId, BufRange>> for SessionImpl<'a, 'b, 'c> {
    fn define_symbol(
        &mut self,
        name: Self::SymbolId,
        _span: RcSpan<BufId, BufRange>,
        expr: Expr<Self::SymbolId, RcSpan<BufId, BufRange>>,
    ) {
        let location = self.builder.context.vars.alloc();
        self.builder.push(Fragment::Reloc(location));
        self.builder.context.content.symbols.define(
            name.content().unwrap(),
            ContentDef::Expr(ExprDef { expr, location }),
        );
    }

    fn emit_fragment(&mut self, fragment: Fragment<Expr<Self::SymbolId, RcSpan<BufId, BufRange>>>) {
        self.builder.push(fragment)
    }

    fn is_non_zero(
        &mut self,
        value: Expr<Self::SymbolId, RcSpan<BufId, BufRange>>,
    ) -> Option<bool> {
        value
            .to_num(&self.builder.context, &mut self.diagnostics)
            .exact()
            .map(|n| n != 0)
    }

    fn set_origin(&mut self, addr: Expr<Self::SymbolId, RcSpan<BufId, BufRange>>) {
        match self.builder.state.take().unwrap() {
            BuilderState::SectionPrelude(index) => {
                self.builder.context.content.sections[index]
                    .constraints
                    .addr = Some(addr);
                self.builder.state = Some(BuilderState::SectionPrelude(index))
            }
            _ => self.builder.state = Some(BuilderState::AnonSectionPrelude { addr: Some(addr) }),
        }
    }

    fn start_section(&mut self, name: SymbolId, _: RcSpan<BufId, BufRange>) {
        let index = self.builder.context.content.sections.len();
        self.builder.state = Some(BuilderState::SectionPrelude(index));
        self.builder.add_section(Some(name.content().unwrap()))
    }
}

impl<'a, R, N, D, S> Backend<S> for CompositeSession<R, N, ObjectBuilder<'a, S>, D>
where
    D: Diagnostics<S>,
    S: Clone,
{
    fn define_symbol(&mut self, name: Self::SymbolId, _span: S, expr: Expr<Self::SymbolId, S>) {
        let location = self.builder.context.vars.alloc();
        self.builder.push(Fragment::Reloc(location));
        self.builder.context.content.symbols.define(
            name.content().unwrap(),
            ContentDef::Expr(ExprDef { expr, location }),
        );
    }

    fn emit_fragment(&mut self, fragment: Fragment<Expr<Self::SymbolId, S>>) {
        self.builder.push(fragment)
    }

    fn is_non_zero(&mut self, value: Expr<Self::SymbolId, S>) -> Option<bool> {
        value
            .to_num(&self.builder.context, &mut self.diagnostics)
            .exact()
            .map(|n| n != 0)
    }

    fn set_origin(&mut self, addr: Expr<Self::SymbolId, S>) {
        match self.builder.state.take().unwrap() {
            BuilderState::SectionPrelude(index) => {
                self.builder.context.content.sections[index]
                    .constraints
                    .addr = Some(addr);
                self.builder.state = Some(BuilderState::SectionPrelude(index))
            }
            _ => self.builder.state = Some(BuilderState::AnonSectionPrelude { addr: Some(addr) }),
        }
    }

    fn start_section(&mut self, name: SymbolId, _: S) {
        let index = self.builder.context.content.sections.len();
        self.builder.state = Some(BuilderState::SectionPrelude(index));
        self.builder.add_section(Some(name.content().unwrap()))
    }
}

impl<'a, R, N, B, D, Span> AllocSymbol<Span> for CompositeSession<R, N, B, D>
where
    Self: SymbolSource<SymbolId = B::SymbolId>,
    B: AllocSymbol<Span>,
    Span: Clone,
{
    fn alloc_symbol(&mut self, span: Span) -> Self::SymbolId {
        self.builder.alloc_symbol(span)
    }
}

impl<'a, S: Clone> SymbolSource for ObjectBuilder<'a, S> {
    type SymbolId = SymbolId;
}

impl<'a, S: Clone> AllocSymbol<S> for ObjectBuilder<'a, S> {
    fn alloc_symbol(&mut self, _span: S) -> Self::SymbolId {
        self.context.content.symbols.alloc().into()
    }
}

impl<'a, S: Clone> BuiltinSymbols for ObjectBuilder<'a, S> {
    type Name = SymbolId;

    fn builtin_symbols(&self) -> &[(&str, Self::Name)] {
        crate::object::eval::BUILTIN_SYMBOLS
    }
}

#[cfg(test)]
pub mod mock {
    use super::*;

    use crate::diag::span::{Source, Spanned};
    use crate::expr::{Atom, Expr, ExprOp};
    use crate::log::Log;

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(crate) struct MockSymbolId(pub usize);

    pub(crate) struct MockBackend<A, T> {
        alloc: A,
        pub log: Log<T>,
    }

    #[derive(Debug, PartialEq)]
    pub enum BackendEvent<N, V: Source> {
        EmitFragment(Fragment<V>),
        SetOrigin(V),
        DefineSymbol((N, V::Span), V),
        StartSection(N, V::Span),
    }

    impl<A, T> MockBackend<A, T> {
        pub fn new(alloc: A, log: Log<T>) -> Self {
            MockBackend { alloc, log }
        }
    }

    impl From<usize> for Atom<usize> {
        fn from(n: usize) -> Self {
            Atom::Name(n)
        }
    }

    impl<A: SymbolSource, T> SymbolSource for MockBackend<A, T> {
        type SymbolId = A::SymbolId;
    }

    impl<A: AllocSymbol<S>, T, S: Clone> AllocSymbol<S> for MockBackend<A, T> {
        fn alloc_symbol(&mut self, span: S) -> Self::SymbolId {
            self.alloc.alloc_symbol(span)
        }
    }

    impl<R, N, D, A, T, S> Backend<S> for CompositeSession<R, N, MockBackend<A, T>, D>
    where
        A: AllocSymbol<S>,
        T: From<BackendEvent<A::SymbolId, Expr<A::SymbolId, S>>>,
        S: Clone,
    {
        fn define_symbol(&mut self, name: Self::SymbolId, span: S, expr: Expr<Self::SymbolId, S>) {
            self.builder
                .log
                .push(BackendEvent::DefineSymbol((name, span), expr));
        }

        fn emit_fragment(&mut self, fragment: Fragment<Expr<Self::SymbolId, S>>) {
            self.builder.log.push(BackendEvent::EmitFragment(fragment))
        }

        fn is_non_zero(&mut self, value: Expr<Self::SymbolId, S>) -> Option<bool> {
            match value.0.as_slice() {
                [Spanned {
                    item: ExprOp::Atom(Atom::Const(n)),
                    ..
                }] => Some(*n != 0),
                _ => None,
            }
        }

        fn set_origin(&mut self, origin: Expr<Self::SymbolId, S>) {
            self.builder.log.push(BackendEvent::SetOrigin(origin))
        }

        fn start_section(&mut self, name: A::SymbolId, span: S) {
            self.builder
                .log
                .push(BackendEvent::StartSection(name, span))
        }
    }

    pub struct SerialIdAllocator<T>(usize, fn(usize) -> T);

    impl<T> SerialIdAllocator<T> {
        pub fn new(wrapper: fn(usize) -> T) -> Self {
            Self(0, wrapper)
        }

        pub fn gen(&mut self) -> T {
            let id = self.0;
            self.0 += 1;
            (self.1)(id)
        }
    }

    impl<T: Clone> SymbolSource for SerialIdAllocator<T> {
        type SymbolId = T;
    }

    impl<T: Clone, S: Clone> AllocSymbol<S> for SerialIdAllocator<T> {
        fn alloc_symbol(&mut self, _: S) -> Self::SymbolId {
            self.gen()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::diag::span::WithSpan;
    use crate::diag::*;
    use crate::expr::{Atom, BinOp, ExprOp};
    use crate::link::Program;
    use crate::object::SectionId;

    use std::borrow::Borrow;

    #[test]
    fn new_object_has_no_sections() {
        let object = build_object::<_, ()>(|_| ());
        assert_eq!(object.content.sections.len(), 0)
    }

    #[test]
    fn no_origin_by_default() {
        let object =
            build_object::<_, ()>(|mut session| session.emit_fragment(Fragment::Byte(0x00)));
        assert_eq!(object.content.sections[0].constraints.addr, None)
    }

    #[test]
    fn constrain_origin_determines_origin_of_new_section() {
        let origin: Expr<_, _> = 0x3000.into();
        let object = build_object(|mut session| {
            session.set_origin(origin.clone());
            session.emit_fragment(Fragment::Byte(0x00))
        });
        assert_eq!(object.content.sections[0].constraints.addr, Some(origin))
    }

    #[test]
    fn start_section_adds_named_section() {
        let mut wrapped_name = None;
        let object = build_object(|mut session| {
            let name = session.alloc_symbol(());
            session.start_section(name, ());
            wrapped_name = Some(name);
        });
        assert_eq!(
            object
                .content
                .symbols
                .get(wrapped_name.unwrap().content().unwrap()),
            Some(&ContentDef::Section(SectionId(0)))
        )
    }

    #[test]
    fn set_origin_in_section_prelude_sets_origin() {
        let origin: Expr<_, _> = 0x0150.into();
        let object = build_object(|mut session| {
            let name = session.alloc_symbol(());
            session.start_section(name, ());
            session.set_origin(origin.clone())
        });
        assert_eq!(object.content.sections[0].constraints.addr, Some(origin))
    }

    #[test]
    fn emit_fragment_into_named_section() {
        let object = build_object(|mut session| {
            let name = session.alloc_symbol(());
            session.start_section(name, ());
            session.emit_fragment(Fragment::Byte(0x00))
        });
        assert_eq!(object.content.sections[0].fragments, [Fragment::Byte(0x00)])
    }

    fn build_object<F: FnOnce(Session<S>), S>(f: F) -> Object<S> {
        let mut linkable = Object::new();
        let session = CompositeSession {
            reentrancy: (),
            names: (),
            builder: ObjectBuilder::new(&mut linkable),
            diagnostics: TestDiagnosticsListener::new(),
        };
        f(session);
        linkable
    }

    type Session<'a, S> =
        CompositeSession<(), (), ObjectBuilder<'a, S>, TestDiagnosticsListener<S>>;

    fn emit_items_and_compare<I, B>(items: I, bytes: B)
    where
        I: Borrow<[Fragment<Expr<SymbolId, ()>>]>,
        B: Borrow<[u8]>,
    {
        let (object, _) = with_object_builder(|mut builder| {
            for item in items.borrow() {
                builder.emit_fragment(item.clone())
            }
        });
        assert_eq!(object.sections.last().unwrap().data, bytes.borrow())
    }

    #[test]
    fn emit_literal_byte_item() {
        emit_items_and_compare([byte_literal(0xff)], [0xff])
    }

    #[test]
    fn emit_two_literal_byte_item() {
        emit_items_and_compare([byte_literal(0x12), byte_literal(0x34)], [0x12, 0x34])
    }

    fn byte_literal(value: i32) -> Fragment<Expr<SymbolId, ()>> {
        Fragment::Immediate(value.into(), Width::Byte)
    }

    #[test]
    fn emit_diagnostic_when_byte_item_out_of_range() {
        test_diagnostic_for_out_of_range_byte(i32::from(i8::min_value()) - 1);
        test_diagnostic_for_out_of_range_byte(i32::from(u8::max_value()) + 1)
    }

    fn test_diagnostic_for_out_of_range_byte(value: i32) {
        let (_, diagnostics) =
            with_object_builder(|mut builder| builder.emit_fragment(byte_literal(value)));
        assert_eq!(
            *diagnostics,
            [Message::ValueOutOfRange {
                value,
                width: Width::Byte,
            }
            .at(())
            .into()]
        );
    }

    #[test]
    fn diagnose_unresolved_symbol() {
        let name = "ident";
        let (_, diagnostics) = with_object_builder(|mut builder| {
            let symbol_id = builder.alloc_symbol(name.into());
            builder.emit_fragment(word_item(Expr(vec![
                ExprOp::Atom(Atom::Name(symbol_id)).with_span(name.into())
            ])))
        });
        assert_eq!(*diagnostics, [unresolved(name)]);
    }

    #[test]
    fn diagnose_two_unresolved_symbols_in_one_expr() {
        let name1 = "ident1";
        let name2 = "ident2";
        let (_, diagnostics) = with_object_builder(|mut builder| {
            let id1 = builder.alloc_symbol(name1.into());
            let id2 = builder.alloc_symbol(name2.into());
            builder.emit_fragment(word_item(Expr(vec![
                ExprOp::Atom(Atom::Name(id1)).with_span(name1.into()),
                ExprOp::Atom(Atom::Name(id2)).with_span(name2.into()),
                ExprOp::Binary(BinOp::Minus).with_span("diff".into()),
            ])))
        });
        assert_eq!(*diagnostics, [unresolved(name1), unresolved(name2)]);
    }

    #[test]
    fn emit_defined_symbol() {
        let (object, diagnostics) = with_object_builder(|mut builder| {
            let symbol_id = builder.alloc_symbol(());
            builder.define_symbol(
                symbol_id,
                (),
                Expr(vec![ExprOp::Atom(Atom::Location).with_span(())]),
            );
            builder.emit_fragment(word_item(Expr(vec![
                ExprOp::Atom(Atom::Name(symbol_id)).with_span(())
            ])));
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x00, 0x00])
    }

    #[test]
    fn emit_symbol_defined_after_use() {
        let (object, diagnostics) = with_object_builder(|mut builder| {
            let symbol_id = builder.alloc_symbol(());
            builder.emit_fragment(word_item(Expr(vec![
                ExprOp::Atom(Atom::Name(symbol_id)).with_span(())
            ])));
            builder.define_symbol(
                symbol_id,
                (),
                Expr(vec![ExprOp::Atom(Atom::Location).with_span(())]),
            );
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x02, 0x00])
    }

    #[test]
    fn reserve_bytes_in_section() {
        let bytes = 3;
        let program =
            build_object(|mut builder| builder.emit_fragment(Fragment::Reserved(bytes.into())));
        assert_eq!(
            program.content.sections[0].fragments,
            [Fragment::Reserved(bytes.into())]
        )
    }

    #[test]
    fn eval_zero() {
        build_object(|mut object_builder| {
            assert_eq!(
                object_builder.is_non_zero(Expr(vec![ExprOp::Atom(Atom::Const(0)).with_span(())])),
                Some(false)
            )
        });
    }

    #[test]
    fn eval_42() {
        build_object(|mut object_builder| {
            assert_eq!(
                object_builder.is_non_zero(Expr(vec![ExprOp::Atom(Atom::Const(42)).with_span(())])),
                Some(true)
            )
        });
    }

    fn with_object_builder<S, F>(f: F) -> (Program, Box<[CompactDiag<S, S>]>)
    where
        S: Clone + 'static,
        F: FnOnce(Session<S>),
    {
        let mut diagnostics = TestDiagnosticsListener::new();
        let object = Program::link(build_object(f), &mut diagnostics);
        let diagnostics = diagnostics.diagnostics.into_inner().into_boxed_slice();
        (object, diagnostics)
    }

    fn word_item<S: Clone>(value: Expr<SymbolId, S>) -> Fragment<Expr<SymbolId, S>> {
        Fragment::Immediate(value, Width::Word)
    }

    fn unresolved(symbol: impl Into<String>) -> CompactDiag<String, String> {
        let symbol = symbol.into();
        Message::UnresolvedSymbol {
            symbol: symbol.clone(),
        }
        .at(symbol)
        .into()
    }
}
