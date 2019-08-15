use self::lowering::Lower;

use super::*;

use crate::diag::span::{Source, WithSpan};
use crate::expr::{BinOp, Expr, ExprOp, FnCall, LocationCounter, ParamId};
use crate::BuiltinSymbols;

mod lowering;

pub trait Backend<S: Clone>: PartialBackend<S> + Sized {
    type ConstBuilder: ValueBuilder<Self::Name, S, Parent = Self, Value = Self::Value>;
    type SymbolBuilder: ValueBuilder<Self::Name, S, Parent = Self, Value = ()>;

    fn build_const(self) -> Self::ConstBuilder;
    fn define_symbol(self, name: Self::Name, span: S) -> Self::SymbolBuilder;
}

pub trait PartialBackend<S: Clone>: AllocName<S> {
    type Value: Source<Span = S>;

    fn emit_item(&mut self, item: Item<Self::Value>);
    fn reserve(&mut self, bytes: Self::Value);
    fn set_origin(&mut self, origin: Self::Value);
    fn start_section(&mut self, name: Self::Name, span: S);
}

pub trait AllocName<S: Clone> {
    type Name: Clone;

    fn alloc_name(&mut self, span: S) -> Self::Name;
}

pub trait ValueBuilder<N: Clone, S: Clone>:
    AllocName<S, Name = N>
    + PushOp<LocationCounter, S>
    + PushOp<i32, S>
    + PushOp<Name<N>, S>
    + PushOp<BinOp, S>
    + PushOp<ParamId, S>
    + PushOp<FnCall, S>
    + Finish
{
}

impl<T, N: Clone, S: Clone> ValueBuilder<N, S> for T where
    Self: AllocName<S, Name = N>
        + PushOp<LocationCounter, S>
        + PushOp<i32, S>
        + PushOp<Name<N>, S>
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

    fn finish(self) -> (Self::Parent, Self::Value);
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
    AnonSectionPrelude { addr: Option<Const<S>> },
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

    fn push(&mut self, node: Node<S>) {
        self.current_section().items.push(node)
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

    fn add_section(&mut self, symbol: Option<ContentSymbol>) {
        self.context.content.add_section(
            symbol,
            self.context.vars.alloc(),
            self.context.vars.alloc(),
        )
    }
}

impl<'a, S: Clone> Backend<S> for ObjectBuilder<'a, S> {
    type ConstBuilder = RelocContext<Self, Const<S>>;
    type SymbolBuilder = SymbolBuilder<'a, S>;

    fn build_const(self) -> Self::ConstBuilder {
        RelocContext::new(self)
    }

    fn define_symbol(self, name: Self::Name, span: S) -> Self::SymbolBuilder {
        let location = self.context.vars.alloc();
        SymbolBuilder {
            parent: self,
            location,
            name: (name.content().unwrap(), span),
            formula: Default::default(),
        }
    }
}

impl<'a, S: Clone> PartialBackend<S> for ObjectBuilder<'a, S> {
    type Value = Const<S>;

    fn emit_item(&mut self, item: Item<Self::Value>) {
        item.lower().for_each(|data_item| self.push(data_item))
    }

    fn reserve(&mut self, bytes: Self::Value) {
        self.current_section().items.push(Node::Reserved(bytes))
    }

    fn set_origin(&mut self, addr: Self::Value) {
        match self.state.take().unwrap() {
            BuilderState::SectionPrelude(index) => {
                self.context.content.sections[index].constraints.addr = Some(addr);
                self.state = Some(BuilderState::SectionPrelude(index))
            }
            _ => self.state = Some(BuilderState::AnonSectionPrelude { addr: Some(addr) }),
        }
    }

    fn start_section(&mut self, name: Symbol, _: S) {
        let index = self.context.content.sections.len();
        self.state = Some(BuilderState::SectionPrelude(index));
        self.add_section(Some(name.content().unwrap()))
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

impl<P, N, S: Clone> PushOp<Name<N>, S> for RelocContext<P, Expr<Atom<LocationCounter, N>, S>> {
    fn push_op(&mut self, name: Name<N>, span: S) {
        self.builder.push_op(name, span)
    }
}

impl<'a, S: Clone> AllocName<S> for RelocContext<ObjectBuilder<'a, S>, Const<S>> {
    type Name = Symbol;

    fn alloc_name(&mut self, span: S) -> Self::Name {
        self.parent.alloc_name(span)
    }
}

impl<'a, S: Clone> Finish for RelocContext<ObjectBuilder<'a, S>, Const<S>> {
    type Parent = ObjectBuilder<'a, S>;
    type Value = Const<S>;

    fn finish(self) -> (Self::Parent, Self::Value) {
        (self.parent, self.builder)
    }
}

pub(crate) struct SymbolBuilder<'a, S> {
    parent: ObjectBuilder<'a, S>,
    location: VarId,
    name: (ContentSymbol, S),
    formula: Formula<S>,
}

impl<'a, S: Clone> AllocName<S> for SymbolBuilder<'a, S> {
    type Name = Symbol;

    fn alloc_name(&mut self, span: S) -> Self::Name {
        self.parent.alloc_name(span)
    }
}

macro_rules! impl_push_op_for_symbol_builder {
    ($t:ty) => {
        impl<'a, S: Clone> PushOp<$t, S> for SymbolBuilder<'a, S> {
            fn push_op(&mut self, op: $t, span: S) {
                self.formula.push_op(op, span)
            }
        }
    };
}

impl_push_op_for_symbol_builder! {i32}
impl_push_op_for_symbol_builder! {Name<Symbol>}
impl_push_op_for_symbol_builder! {ParamId}
impl_push_op_for_symbol_builder! {BinOp}
impl_push_op_for_symbol_builder! {FnCall}

impl<'a, S: Clone> PushOp<LocationCounter, S> for SymbolBuilder<'a, S> {
    fn push_op(&mut self, _: LocationCounter, span: S) {
        self.formula
            .0
            .push(ExprOp::Atom(Atom::Location(self.location)).with_span(span))
    }
}

impl<'a, S> Finish for SymbolBuilder<'a, S> {
    type Parent = ObjectBuilder<'a, S>;
    type Value = ();

    fn finish(self) -> (Self::Parent, Self::Value) {
        let mut parent = self.parent;
        parent.push(Node::Reloc(self.location));
        parent
            .context
            .content
            .symbols
            .define(self.name.0, ContentDef::Formula(self.formula));
        (parent, ())
    }
}

impl<'a, S: Clone> AllocName<S> for ObjectBuilder<'a, S> {
    type Name = Symbol;

    fn alloc_name(&mut self, _span: S) -> Self::Name {
        self.context.content.symbols.alloc().into()
    }
}

impl<'a, S: Clone> BuiltinSymbols for ObjectBuilder<'a, S> {
    type Name = Symbol;

    fn builtin_symbols(&self) -> &[(&str, Self::Name)] {
        super::eval::BUILTIN_SYMBOLS
    }
}

impl<L, N> From<Name<N>> for Atom<L, N> {
    fn from(Name(name): Name<N>) -> Self {
        Atom::Name(name)
    }
}

impl<T: Into<ExprOp<A>>, A, S: Clone> PushOp<T, S> for Expr<A, S> {
    fn push_op(&mut self, op: T, span: S) {
        self.0.push(op.into().with_span(span))
    }
}

impl<N, A: From<Name<N>>> From<Name<N>> for ExprOp<A> {
    fn from(name: Name<N>) -> Self {
        ExprOp::Atom(name.into())
    }
}

#[cfg(test)]
pub mod mock {
    use super::*;

    use crate::expr::Atom;
    use crate::log::Log;

    use std::marker::PhantomData;

    type Expr<N, S> = crate::expr::Expr<Atom<LocationCounter, N>, S>;

    pub(crate) struct MockBackend<A, T> {
        alloc: A,
        pub log: Log<T>,
    }

    #[derive(Debug, PartialEq)]
    pub enum BackendEvent<N, V: Source> {
        EmitItem(Item<V>),
        Reserve(V),
        SetOrigin(V),
        DefineSymbol((N, V::Span), V),
        StartSection(N, V::Span),
    }

    impl<A, T> MockBackend<A, T> {
        pub fn new(alloc: A, log: Log<T>) -> Self {
            MockBackend { alloc, log }
        }
    }

    impl<A, T, S> Backend<S> for MockBackend<A, T>
    where
        A: AllocName<S>,
        T: From<BackendEvent<A::Name, Expr<A::Name, S>>>,
        S: Clone,
    {
        type ConstBuilder = RelocContext<Self, Expr<A::Name, S>>;
        type SymbolBuilder = SymbolBuilder<A, T, S>;

        fn build_const(self) -> Self::ConstBuilder {
            RelocContext::new(self)
        }

        fn define_symbol(self, name: Self::Name, span: S) -> Self::SymbolBuilder {
            RelocContext::new((self, (name, span)))
        }
    }

    type SymbolBuilder<A, T, S> = RelocContext<
        (MockBackend<A, T>, (<A as AllocName<S>>::Name, S)),
        Expr<<A as AllocName<S>>::Name, S>,
    >;

    impl<A, T, S> AllocName<S> for RelocContext<MockBackend<A, T>, Expr<A::Name, S>>
    where
        A: AllocName<S>,
        S: Clone,
    {
        type Name = A::Name;

        fn alloc_name(&mut self, span: S) -> Self::Name {
            self.parent.alloc_name(span)
        }
    }

    impl<A: AllocName<S>, T, S: Clone> AllocName<S> for SymbolBuilder<A, T, S> {
        type Name = A::Name;

        fn alloc_name(&mut self, span: S) -> Self::Name {
            self.parent.0.alloc_name(span)
        }
    }

    impl<A, T, S> Finish for RelocContext<MockBackend<A, T>, Expr<A::Name, S>>
    where
        A: AllocName<S>,
        S: Clone,
    {
        type Parent = MockBackend<A, T>;
        type Value = Expr<A::Name, S>;

        fn finish(self) -> (Self::Parent, Self::Value) {
            (self.parent, self.builder)
        }
    }

    impl<A, T, S> Finish for SymbolBuilder<A, T, S>
    where
        A: AllocName<S>,
        T: From<BackendEvent<A::Name, Expr<A::Name, S>>>,
        S: Clone,
    {
        type Parent = MockBackend<A, T>;
        type Value = ();

        fn finish(self) -> (Self::Parent, Self::Value) {
            let (parent, name) = self.parent;
            parent
                .log
                .push(BackendEvent::DefineSymbol(name, self.builder));
            (parent, ())
        }
    }

    impl<L> From<usize> for Atom<L, usize> {
        fn from(n: usize) -> Self {
            Atom::Name(n)
        }
    }

    impl<A: AllocName<S>, T, S: Clone> AllocName<S> for MockBackend<A, T> {
        type Name = A::Name;

        fn alloc_name(&mut self, span: S) -> Self::Name {
            self.alloc.alloc_name(span)
        }
    }

    impl<A, T, S> PartialBackend<S> for MockBackend<A, T>
    where
        A: AllocName<S>,
        T: From<BackendEvent<A::Name, Expr<A::Name, S>>>,
        S: Clone,
    {
        type Value = Expr<A::Name, S>;

        fn emit_item(&mut self, item: Item<Self::Value>) {
            self.log.push(BackendEvent::EmitItem(item))
        }

        fn reserve(&mut self, bytes: Self::Value) {
            self.log.push(BackendEvent::Reserve(bytes))
        }

        fn set_origin(&mut self, origin: Self::Value) {
            self.log.push(BackendEvent::SetOrigin(origin))
        }

        fn start_section(&mut self, name: A::Name, span: S) {
            self.log.push(BackendEvent::StartSection(name, span))
        }
    }

    pub struct SerialIdAllocator(usize);

    impl SerialIdAllocator {
        pub fn new() -> Self {
            Self(0)
        }

        pub fn gen(&mut self) -> usize {
            let id = self.0;
            self.0 += 1;
            id
        }
    }

    impl<S: Clone> AllocName<S> for SerialIdAllocator {
        type Name = usize;

        fn alloc_name(&mut self, _: S) -> Self::Name {
            self.gen()
        }
    }

    pub struct PanickingIdAllocator<I>(PhantomData<I>);

    impl<I> PanickingIdAllocator<I> {
        pub fn new() -> Self {
            Self(PhantomData)
        }
    }

    impl<I: Clone, S: Clone> AllocName<S> for PanickingIdAllocator<I> {
        type Name = I;

        fn alloc_name(&mut self, _: S) -> Self::Name {
            panic!("tried to allocate an ID")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::diag::{CompactDiag, Message, TestDiagnosticsListener};
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
        let object = build_object::<_, ()>(|mut builder| builder.push(Node::Byte(0xcd)));
        assert_eq!(object.content.sections[0].constraints.addr, None)
    }

    #[test]
    fn constrain_origin_determines_origin_of_new_section() {
        let origin: Const<_> = 0x3000.into();
        let object = build_object(|mut builder| {
            builder.set_origin(origin.clone());
            builder.push(Node::Byte(0xcd))
        });
        assert_eq!(object.content.sections[0].constraints.addr, Some(origin))
    }

    #[test]
    fn start_section_adds_named_section() {
        let mut wrapped_name = None;
        let object = build_object(|mut builder| {
            let name = builder.alloc_name(());
            builder.start_section(name, ());
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
        let origin: Const<_> = 0x0150.into();
        let object = build_object(|mut builder| {
            let name = builder.alloc_name(());
            builder.start_section(name, ());
            builder.set_origin(origin.clone())
        });
        assert_eq!(object.content.sections[0].constraints.addr, Some(origin))
    }

    #[test]
    fn push_node_into_named_section() {
        let node = Node::Byte(0x42);
        let object = build_object(|mut builder| {
            let name = builder.alloc_name(());
            builder.start_section(name, ());
            builder.push(node.clone())
        });
        assert_eq!(object.content.sections[0].items, [node])
    }

    fn build_object<F: FnOnce(ObjectBuilder<S>), S>(f: F) -> Object<S> {
        let mut linkable = Object::new();
        let builder = ObjectBuilder::new(&mut linkable);
        f(builder);
        linkable
    }

    #[test]
    fn emit_stop() {
        emit_items_and_compare(
            [Item::CpuInstr(CpuInstr::Nullary(Nullary::Stop))],
            [0x10, 0x00],
        )
    }

    fn emit_items_and_compare<I, B>(items: I, bytes: B)
    where
        I: Borrow<[Item<Const<()>>]>,
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

    fn byte_literal(value: i32) -> Item<Const<()>> {
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
            let symbol_id = builder.alloc_name(name.into());
            let mut value: Const<_> = Default::default();
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
                let id1 = builder.alloc_name(name1.into());
                let mut value: Const<_> = Default::default();
                value.push_op(id1, name1.into());
                let id2 = builder.alloc_name(name2.into());
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
            let symbol_id = builder.alloc_name(());
            let mut builder = builder.define_symbol(symbol_id, ());
            builder.push_op(LocationCounter, ());
            let (mut builder, _) = builder.finish();
            let mut value: Const<_> = Default::default();
            value.push_op(symbol_id, ());
            builder.emit_item(word_item(value));
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x00, 0x00])
    }

    #[test]
    fn emit_symbol_defined_after_use() {
        let (object, diagnostics) = with_object_builder(|mut builder| {
            let symbol_id = builder.alloc_name(());
            let mut value: Const<_> = Default::default();
            value.push_op(symbol_id, ());
            builder.emit_item(word_item(value));
            let mut builder = builder.define_symbol(symbol_id, ());
            builder.push_op(LocationCounter, ());
            builder.finish();
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x02, 0x00])
    }

    #[test]
    fn reserve_bytes_in_section() {
        let bytes = 3;
        let program = build_object(|mut builder| builder.reserve(bytes.into()));
        assert_eq!(
            program.content.sections[0].items,
            [Node::Reserved(bytes.into())]
        )
    }

    fn with_object_builder<S, F>(f: F) -> (Program, Box<[CompactDiag<S, S>]>)
    where
        S: Clone + 'static,
        F: FnOnce(ObjectBuilder<S>),
    {
        let mut diagnostics = TestDiagnosticsListener::new();
        let object = Program::link(build_object(f), &mut diagnostics);
        let diagnostics = diagnostics.diagnostics.into_inner().into_boxed_slice();
        (object, diagnostics)
    }

    fn word_item<S: Clone>(value: Const<S>) -> Item<Const<S>> {
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
