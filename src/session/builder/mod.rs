use self::lowering::Lower;

use crate::diag::span::{Source, WithSpan};
use crate::diag::Diagnostics;
use crate::expr::{Atom, BinOp, ExprOp, FnCall, LocationCounter, ParamId};
use crate::object::*;
use crate::{BuiltinSymbols, CompositeSession};

mod lowering;

pub(crate) trait Backend<S: Clone>: PartialBackend<S> + Sized {
    type ExprBuilder: ValueBuilder<S, SymbolId = Self::SymbolId, Parent = Self, Value = Self::Value>
        + Diagnostics<S>;

    fn build_const(self) -> Self::ExprBuilder;
    fn define_symbol(&mut self, name: Self::SymbolId, span: S, expr: Self::Value);
}

pub(crate) trait PartialBackend<S: Clone>: AllocSymbol<S> {
    type Value: Source<Span = S>;

    fn emit_fragment(&mut self, fragment: Fragment<Self::Value>);
    fn emit_item(&mut self, item: Item<Self::Value>);
    fn is_non_zero(&mut self, value: Self::Value) -> Option<bool>;
    fn set_origin(&mut self, origin: Self::Value);
    fn start_section(&mut self, name: Self::SymbolId, span: S);
}

pub trait AllocSymbol<S: Clone>: SymbolSource {
    fn alloc_symbol(&mut self, span: S) -> Self::SymbolId;
}

pub trait SymbolSource {
    type SymbolId: Clone;
}

pub trait ValueBuilder<S: Clone>:
    AllocSymbol<S>
    + PushOp<LocationCounter, S>
    + PushOp<i32, S>
    + PushOp<Name<<Self as SymbolSource>::SymbolId>, S>
    + PushOp<BinOp, S>
    + PushOp<ParamId, S>
    + PushOp<FnCall, S>
    + Finish
{
}

impl<T, S: Clone> ValueBuilder<S> for T where
    Self: AllocSymbol<S>
        + PushOp<LocationCounter, S>
        + PushOp<i32, S>
        + PushOp<Name<<Self as SymbolSource>::SymbolId>, S>
        + PushOp<BinOp, S>
        + PushOp<ParamId, S>
        + PushOp<FnCall, S>
        + Finish
{
}

pub trait PushOp<T, S: Clone> {
    fn push_op(&mut self, op: T, span: S);
}

#[derive(Clone)]
pub struct Name<T>(pub T);

pub trait Finish {
    type Parent;
    type Value;

    fn finish(self) -> (Self::Parent, Option<Self::Value>);
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item<V: Source> {
    CpuInstr(CpuInstr<V>),
    Data(V, Width),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Width {
    Byte,
    Word,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CpuInstr<V: Source> {
    AddHl(Reg16),
    Alu(AluOperation, AluSource<V>),
    Bit(BitOperation, V, SimpleOperand),
    IncDec8(IncDec, SimpleOperand),
    IncDec16(IncDec, Reg16),
    JpDerefHl,
    Branch(Branch<V>, Option<Condition>),
    Ld(Ld<V>),
    Ldhl(V),
    Misc(MiscOperation, SimpleOperand),
    Nullary(Nullary),
    Pop(RegPair),
    Push(RegPair),
    Rst(V),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Nullary {
    Cpl,
    Daa,
    Di,
    Ei,
    Halt,
    Nop,
    Reti,
    Rla,
    Rlca,
    Rra,
    Rrca,
    Stop,
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

#[derive(Clone, Debug, PartialEq)]
pub enum AluSource<V> {
    Simple(SimpleOperand),
    Immediate(V),
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
pub enum SimpleOperand {
    A,
    B,
    C,
    D,
    E,
    H,
    L,
    DerefHl,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Ld<V> {
    Simple(SimpleOperand, SimpleOperand),
    Special(SpecialLd<V>, Direction),
    SpHl,
    Immediate8(SimpleOperand, V),
    Immediate16(Reg16, V),
}

#[derive(Clone, Debug, PartialEq)]
pub enum SpecialLd<V> {
    DerefPtrReg(PtrReg),
    InlineAddr(V),
    RegIndex,
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

#[derive(Clone, Debug, PartialEq)]
pub enum Branch<V> {
    Call(V),
    Jp(V),
    Jr(V),
    Ret,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Condition {
    C,
    Nc,
    Nz,
    Z,
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
    AnonSectionPrelude { addr: Option<Expr<S>> },
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

    fn push(&mut self, fragment: Fragment<Expr<S>>) {
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

impl<'a, R, N, S> Backend<S> for CompositeSession<R, N, ObjectBuilder<'a, S>>
where
    R: Diagnostics<S>,
    S: Clone,
{
    type ExprBuilder = CompositeSession<R, N, RelocContext<ObjectBuilder<'a, S>, Expr<S>>>;

    fn build_const(self) -> Self::ExprBuilder {
        CompositeSession {
            reentrancy: self.reentrancy,
            names: self.names,
            builder: RelocContext::new(self.builder),
        }
    }

    fn define_symbol(&mut self, name: Self::SymbolId, _span: S, expr: Self::Value) {
        let location = self.builder.context.vars.alloc();
        self.builder.push(Fragment::Reloc(location));
        self.builder.context.content.symbols.define(
            name.content().unwrap(),
            ContentDef::Expr(ExprDef { expr, location }),
        );
    }
}

impl<'a, R, N, S> PartialBackend<S> for CompositeSession<R, N, ObjectBuilder<'a, S>>
where
    R: Diagnostics<S>,
    S: Clone,
{
    type Value = Expr<S>;

    fn emit_fragment(&mut self, fragment: Fragment<Self::Value>) {
        self.builder.push(fragment)
    }

    fn emit_item(&mut self, item: Item<Self::Value>) {
        item.lower()
            .for_each(|fragment| self.emit_fragment(fragment))
    }

    fn is_non_zero(&mut self, value: Self::Value) -> Option<bool> {
        value
            .to_num(&self.builder.context, &mut self.reentrancy)
            .exact()
            .map(|n| n != 0)
    }

    fn set_origin(&mut self, addr: Self::Value) {
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

impl<'a, R, N, B, Span> AllocSymbol<Span> for CompositeSession<R, N, B>
where
    Self: SymbolSource<SymbolId = B::SymbolId>,
    B: AllocSymbol<Span>,
    Span: Clone,
{
    fn alloc_symbol(&mut self, span: Span) -> Self::SymbolId {
        self.builder.alloc_symbol(span)
    }
}

pub(crate) struct RelocContext<P, B> {
    parent: P,
    builder: B,
}

impl<P, B: Default> RelocContext<P, B> {
    fn new(parent: P) -> Self {
        Self {
            parent,
            builder: Default::default(),
        }
    }
}

macro_rules! impl_push_op_for_reloc_context {
    ($t:ty) => {
        impl<P, B, S> PushOp<$t, S> for RelocContext<P, B>
        where
            B: PushOp<$t, S>,
            S: Clone,
        {
            fn push_op(&mut self, op: $t, span: S) {
                self.builder.push_op(op, span)
            }
        }
    };
}

impl_push_op_for_reloc_context! {LocationCounter}
impl_push_op_for_reloc_context! {i32}
impl_push_op_for_reloc_context! {BinOp}
impl_push_op_for_reloc_context! {ParamId}
impl_push_op_for_reloc_context! {FnCall}

impl<P, N, S: Clone> PushOp<Name<N>, S> for RelocContext<P, crate::expr::Expr<N, S>> {
    fn push_op(&mut self, name: Name<N>, span: S) {
        self.builder.push_op(name, span)
    }
}

impl<R, N, B: PushOp<T, S>, T, S: Clone> PushOp<T, S> for CompositeSession<R, N, B> {
    fn push_op(&mut self, op: T, span: S) {
        self.builder.push_op(op, span)
    }
}

impl<'a, S: Clone> SymbolSource for RelocContext<ObjectBuilder<'a, S>, Expr<S>> {
    type SymbolId = SymbolId;
}

impl<'a, S: Clone> AllocSymbol<S> for RelocContext<ObjectBuilder<'a, S>, Expr<S>> {
    fn alloc_symbol(&mut self, span: S) -> Self::SymbolId {
        self.parent.alloc_symbol(span)
    }
}

impl<R, N, B: Finish> Finish for CompositeSession<R, N, B> {
    type Parent = CompositeSession<R, N, B::Parent>;
    type Value = B::Value;

    fn finish(self) -> (Self::Parent, Option<Self::Value>) {
        let (builder, value) = self.builder.finish();
        (
            CompositeSession {
                reentrancy: self.reentrancy,
                names: self.names,
                builder,
            },
            value,
        )
    }
}

impl<'a, S: Clone> Finish for RelocContext<ObjectBuilder<'a, S>, Expr<S>> {
    type Parent = ObjectBuilder<'a, S>;
    type Value = Expr<S>;

    fn finish(self) -> (Self::Parent, Option<Self::Value>) {
        (
            self.parent,
            if self.builder.0.is_empty() {
                None
            } else {
                Some(self.builder)
            },
        )
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

impl<N> From<Name<N>> for Atom<N> {
    fn from(Name(name): Name<N>) -> Self {
        Atom::Name(name)
    }
}

impl<T: Into<ExprOp<A>>, A, S: Clone> PushOp<T, S> for crate::expr::Expr<A, S> {
    fn push_op(&mut self, op: T, span: S) {
        self.0.push(op.into().with_span(span))
    }
}

impl<N> From<Name<N>> for ExprOp<N> {
    fn from(name: Name<N>) -> Self {
        ExprOp::Atom(name.into())
    }
}

#[cfg(test)]
pub mod mock {
    pub(crate) use super::RelocContext;

    use super::*;

    use crate::diag::span::Spanned;
    use crate::expr::{Atom, Expr};
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
        EmitItem(Item<V>),
        SetOrigin(V),
        DefineSymbol((N, V::Span), V),
        StartSection(N, V::Span),
    }

    impl<A, T> MockBackend<A, T> {
        pub fn new(alloc: A, log: Log<T>) -> Self {
            MockBackend { alloc, log }
        }
    }

    impl<R, N, A, T, S> Backend<S> for CompositeSession<R, N, MockBackend<A, T>>
    where
        R: Diagnostics<S>,
        A: AllocSymbol<S>,
        T: From<BackendEvent<A::SymbolId, Expr<A::SymbolId, S>>>,
        S: Clone,
    {
        type ExprBuilder =
            CompositeSession<R, N, RelocContext<MockBackend<A, T>, Expr<A::SymbolId, S>>>;

        fn build_const(self) -> Self::ExprBuilder {
            CompositeSession {
                reentrancy: self.reentrancy,
                names: self.names,
                builder: RelocContext::new(self.builder),
            }
        }

        fn define_symbol(&mut self, name: Self::SymbolId, span: S, expr: Self::Value) {
            self.builder
                .log
                .push(BackendEvent::DefineSymbol((name, span), expr));
        }
    }

    impl<A, T, S> SymbolSource for RelocContext<MockBackend<A, T>, Expr<A::SymbolId, S>>
    where
        A: SymbolSource,
        S: Clone,
    {
        type SymbolId = A::SymbolId;
    }

    impl<A, T, S> AllocSymbol<S> for RelocContext<MockBackend<A, T>, Expr<A::SymbolId, S>>
    where
        A: AllocSymbol<S>,
        S: Clone,
    {
        fn alloc_symbol(&mut self, span: S) -> Self::SymbolId {
            self.parent.alloc_symbol(span)
        }
    }

    impl<A, T, S> Finish for RelocContext<MockBackend<A, T>, Expr<A::SymbolId, S>>
    where
        A: AllocSymbol<S>,
        S: Clone,
    {
        type Parent = MockBackend<A, T>;
        type Value = Expr<A::SymbolId, S>;

        fn finish(self) -> (Self::Parent, Option<Self::Value>) {
            (
                self.parent,
                if self.builder.0.is_empty() {
                    None
                } else {
                    Some(self.builder)
                },
            )
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

    impl<R, N, A, T, S> PartialBackend<S> for CompositeSession<R, N, MockBackend<A, T>>
    where
        A: AllocSymbol<S>,
        T: From<BackendEvent<A::SymbolId, Expr<A::SymbolId, S>>>,
        S: Clone,
    {
        type Value = Expr<A::SymbolId, S>;

        fn emit_fragment(&mut self, fragment: Fragment<Self::Value>) {
            self.builder.log.push(BackendEvent::EmitFragment(fragment))
        }

        fn emit_item(&mut self, item: Item<Self::Value>) {
            self.builder.log.push(BackendEvent::EmitItem(item))
        }

        fn is_non_zero(&mut self, value: Self::Value) -> Option<bool> {
            match value.0.as_slice() {
                [Spanned {
                    item: ExprOp::Atom(Atom::Const(n)),
                    ..
                }] => Some(*n != 0),
                _ => None,
            }
        }

        fn set_origin(&mut self, origin: Self::Value) {
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

    use crate::diag::*;
    use crate::expr::BinOp;
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
        let object = build_object::<_, ()>(|mut session| {
            session.emit_item(Item::CpuInstr(CpuInstr::Nullary(Nullary::Nop)))
        });
        assert_eq!(object.content.sections[0].constraints.addr, None)
    }

    #[test]
    fn constrain_origin_determines_origin_of_new_section() {
        let origin: Expr<_> = 0x3000.into();
        let object = build_object(|mut session| {
            session.set_origin(origin.clone());
            session.emit_item(Item::CpuInstr(CpuInstr::Nullary(Nullary::Nop)))
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
        let origin: Expr<_> = 0x0150.into();
        let object = build_object(|mut session| {
            let name = session.alloc_symbol(());
            session.start_section(name, ());
            session.set_origin(origin.clone())
        });
        assert_eq!(object.content.sections[0].constraints.addr, Some(origin))
    }

    #[test]
    fn emit_item_into_named_section() {
        let object = build_object(|mut session| {
            let name = session.alloc_symbol(());
            session.start_section(name, ());
            session.emit_item(Item::CpuInstr(CpuInstr::Nullary(Nullary::Nop)))
        });
        assert_eq!(object.content.sections[0].fragments, [Fragment::Byte(0x00)])
    }

    fn build_object<F: FnOnce(Session<S>), S>(f: F) -> Object<S> {
        let mut linkable = Object::new();
        let session = CompositeSession {
            reentrancy: TestDiagnosticsListener::new(),
            names: (),
            builder: ObjectBuilder::new(&mut linkable),
        };
        f(session);
        linkable
    }

    type Session<'a, S> = CompositeSession<TestDiagnosticsListener<S>, (), ObjectBuilder<'a, S>>;

    #[test]
    fn emit_stop() {
        emit_items_and_compare(
            [Item::CpuInstr(CpuInstr::Nullary(Nullary::Stop))],
            [0x10, 0x00],
        )
    }

    fn emit_items_and_compare<I, B>(items: I, bytes: B)
    where
        I: Borrow<[Item<Expr<()>>]>,
        B: Borrow<[u8]>,
    {
        let (object, _) = with_object_builder(|mut builder| {
            for item in items.borrow() {
                builder.emit_item(item.clone())
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

    fn byte_literal(value: i32) -> Item<Expr<()>> {
        Item::Data(value.into(), Width::Byte)
    }

    #[test]
    fn emit_diagnostic_when_byte_item_out_of_range() {
        test_diagnostic_for_out_of_range_byte(i32::from(i8::min_value()) - 1);
        test_diagnostic_for_out_of_range_byte(i32::from(u8::max_value()) + 1)
    }

    fn test_diagnostic_for_out_of_range_byte(value: i32) {
        let (_, diagnostics) =
            with_object_builder(|mut builder| builder.emit_item(byte_literal(value)));
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
            let mut value: Expr<_> = Default::default();
            value.push_op(symbol_id, name.into());
            builder.emit_item(word_item(value))
        });
        assert_eq!(*diagnostics, [unresolved(name)]);
    }

    #[test]
    fn diagnose_two_unresolved_symbols_in_one_expr() {
        let name1 = "ident1";
        let name2 = "ident2";
        let (_, diagnostics) = with_object_builder(|mut builder| {
            let value = {
                let id1 = builder.alloc_symbol(name1.into());
                let mut value: Expr<_> = Default::default();
                value.push_op(id1, name1.into());
                let id2 = builder.alloc_symbol(name2.into());
                value.push_op(id2, name2.into());
                value.push_op(BinOp::Minus, "diff".into());
                value
            };
            builder.emit_item(word_item(value))
        });
        assert_eq!(*diagnostics, [unresolved(name1), unresolved(name2)]);
    }

    #[test]
    fn emit_defined_symbol() {
        let (object, diagnostics) = with_object_builder(|mut builder| {
            let symbol_id = builder.alloc_symbol(());
            let mut builder = builder.build_const();
            builder.push_op(LocationCounter, ());
            let (mut builder, expr) = builder.finish();
            builder.define_symbol(symbol_id, (), expr.unwrap());
            let mut value: Expr<_> = Default::default();
            value.push_op(symbol_id, ());
            builder.emit_item(word_item(value));
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x00, 0x00])
    }

    #[test]
    fn emit_symbol_defined_after_use() {
        let (object, diagnostics) = with_object_builder(|mut builder| {
            let symbol_id = builder.alloc_symbol(());
            let mut value: Expr<_> = Default::default();
            value.push_op(symbol_id, ());
            builder.emit_item(word_item(value));
            let mut builder = builder.build_const();
            builder.push_op(LocationCounter, ());
            let (mut builder, expr) = builder.finish();
            builder.define_symbol(symbol_id, (), expr.unwrap());
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
        build_object(|object_builder| {
            let mut const_builder = object_builder.build_const();
            const_builder.push_op(0, ());
            let (mut object_builder, zero) = const_builder.finish();
            assert_eq!(object_builder.is_non_zero(zero.unwrap(),), Some(false))
        });
    }

    #[test]
    fn eval_42() {
        build_object(|object_builder| {
            let mut const_builder = object_builder.build_const();
            const_builder.push_op(42, ());
            let (mut object_builder, forty_two) = const_builder.finish();
            assert_eq!(object_builder.is_non_zero(forty_two.unwrap(),), Some(true))
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

    fn word_item<S: Clone>(value: Expr<S>) -> Item<Expr<S>> {
        Item::Data(value, Width::Word)
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
