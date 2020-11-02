use crate::codebase::{Codebase, FileSystem, StdFileSystem};
use crate::diagnostics::*;
use crate::object::var::Var;
use crate::object::*;
use crate::program::Program;
use crate::span::SpanSource;
use crate::{Config, DiagnosticsConfig, InputConfig};

use std::borrow::Borrow;

mod translate;

pub struct Linker<'a> {
    config: &'a mut Config<'a>,
}

impl<'a> Linker<'a> {
    pub fn new(config: &'a mut Config<'a>) -> Self {
        Self { config }
    }

    pub fn link<I>(&mut self, objects: I) -> Option<Program>
    where
        I: IntoIterator<Item = Object>,
    {
        let mut input_holder = None;
        let mut diagnostics_holder = None;
        let input: &mut dyn FileSystem = match &mut self.config.input {
            InputConfig::Default => input_holder.get_or_insert_with(StdFileSystem::new),
            InputConfig::Custom(input) => *input,
        };
        let diagnostics: &mut dyn FnMut(Diagnostic) = match &mut self.config.diagnostics {
            DiagnosticsConfig::Ignore => diagnostics_holder.get_or_insert(|_| {}),
            DiagnosticsConfig::Output(diagnostics) => *diagnostics,
        };
        try_link(objects, input, diagnostics)
    }
}

fn try_link<I>(
    objects: I,
    input: &mut dyn FileSystem,
    diagnostics: &mut dyn FnMut(Diagnostic),
) -> Option<Program>
where
    I: IntoIterator<Item = Object>,
{
    let mut session = Session::<_, SpanData>::new(
        input,
        OutputForwarder {
            output: diagnostics,
        },
    );
    for Object(data) in objects {
        session.import_object(data)
    }
    Some(session.link())
}

struct Session<'a, D, M: SpanSource> {
    codebase: Codebase<'a>,
    content: Content<M::Span>,
    diagnostics: D,
    metadata: M,
    var_count: u32,
}

impl<'a, D, M: Default + SpanSource> Session<'a, D, M> {
    fn new(fs: &'a mut dyn FileSystem, diagnostics: D) -> Self {
        Self {
            codebase: Codebase::new(fs),
            content: Content::new(),
            diagnostics,
            metadata: M::default(),
            var_count: 0,
        }
    }

    fn import_object<N>(&mut self, object: ObjectData<N>)
    where
        N: SpanSource<Span = M::Span>,
        Self: ImportMetadata<N>,
        <Self as ImportMetadata<N>>::SpanPatcher: PatchSpan<M::Span>,
    {
        let patcher = self.import_metadata(object.metadata);
        self.import_content(object.content, patcher);
    }

    fn import_content<P: PatchSpan<M::Span>>(&mut self, content: Content<M::Span>, _patcher: P) {
        let base_var = self.var_count;
        let mut max_local_var_id = 0;
        self.content.sections.reserve(content.sections.len());
        for mut section in content.sections {
            max_local_var_id = std::cmp::max(max_local_var_id, section.addr.0);
            section.addr.0 += base_var;
            max_local_var_id = std::cmp::max(max_local_var_id, section.size.0);
            section.size.0 += base_var;
            for fragment in &mut section.fragments {
                if let Fragment::Reloc(VarId(id)) = fragment {
                    max_local_var_id = std::cmp::max(max_local_var_id, *id);
                    *id += base_var;
                }
            }
            self.content.sections.push(section)
        }
        self.content.symbols.0.extend(content.symbols.0);
        self.var_count += max_local_var_id + 1;
    }

    fn link(mut self) -> Program
    where
        for<'r> DiagnosticsContext<'r, 'a, M, D>: Diagnostics<M::Span>,
    {
        let mut vars = VarTable(vec![Var::Unknown; self.var_count as usize]);
        vars.resolve(&self.content);
        let mut context = LinkageContext {
            content: &self.content,
            vars: &vars,
            location: 0.into(),
        };
        let mut diagnostics = DiagnosticsContext {
            codebase: &mut self.codebase,
            registry: &mut self.metadata,
            diagnostics: &mut self.diagnostics,
        };
        Program {
            sections: self
                .content
                .sections()
                .flat_map(|section| section.translate(&mut context, &mut diagnostics))
                .collect(),
        }
    }
}

trait ImportMetadata<M: SpanSource> {
    type SpanPatcher: PatchSpan<M::Span>;
    fn import_metadata(&mut self, metadata: M) -> Self::SpanPatcher;
}

trait PatchSpan<S> {
    fn patch_span(&self, span: &mut S);
}

impl<'a, D> ImportMetadata<Metadata> for Session<'a, D, SpanData> {
    type SpanPatcher = SpanPatcher;

    fn import_metadata(&mut self, metadata: Metadata) -> Self::SpanPatcher {
        for path in Vec::from(metadata.source_files) {
            self.codebase.open(&path).unwrap();
        }
        self.metadata = metadata.span_data;
        SpanPatcher
    }
}

struct SpanPatcher;

impl PatchSpan<Span> for SpanPatcher {
    fn patch_span(&self, _: &mut Span) {
        todo!()
    }
}

impl VarTable {
    fn resolve<S: Clone>(&mut self, content: &Content<S>) {
        self.refine_all(content);
        self.refine_all(content);
    }

    fn refine_all<S: Clone>(&mut self, content: &Content<S>) -> i32 {
        let mut refinements = 0;
        let context = &mut LinkageContext {
            content,
            vars: self,
            location: Var::Unknown,
        };
        for section in content.sections() {
            context.location = section.eval_addr(context);
            context.vars[section.addr].refine(context.location.clone());
            let size = section.traverse(context, |item, context| {
                if let Fragment::Reloc(id) = item {
                    refinements += context.vars[*id].refine(context.location.clone()) as i32
                }
            });
            refinements += context.vars[section.size].refine(size) as i32
        }
        refinements
    }
}

impl<S: Clone> Section<S> {
    fn traverse<V, F>(&self, context: &mut LinkageContext<&Content<S>, V>, mut f: F) -> Var
    where
        V: Borrow<VarTable>,
        F: FnMut(&Fragment<Expr<S>>, &mut LinkageContext<&Content<S>, V>),
    {
        let addr = context.location.clone();
        let mut offset = Var::from(0);
        for item in &self.fragments {
            offset += &item.size(&context);
            context.location = &addr + &offset;
            f(item, context)
        }
        offset
    }

    fn eval_addr<'a, V: Borrow<VarTable>>(
        &self,
        context: &LinkageContext<&'a Content<S>, V>,
    ) -> Var {
        self.constraints
            .addr
            .as_ref()
            .map(|expr| expr.to_num(context, &mut IgnoreDiagnostics))
            .unwrap_or_else(|| 0.into())
    }
}

impl<S: Clone> Fragment<Expr<S>> {
    fn size<'a, V: Borrow<VarTable>>(&self, context: &LinkageContext<&'a Content<S>, V>) -> Var {
        match self {
            Fragment::Byte(_) | Fragment::Embedded(..) => 1.into(),
            Fragment::Immediate(_, width) => width.len().into(),
            Fragment::LdInlineAddr(_, expr) => match expr.to_num(context, &mut IgnoreDiagnostics) {
                Var::Range { min, .. } if min >= 0xff00 => 2.into(),
                Var::Range { max, .. } if max < 0xff00 => 3.into(),
                _ => Var::Range { min: 2, max: 3 },
            },
            Fragment::Reloc(_) => 0.into(),
            Fragment::Reserved(bytes) => bytes.to_num(context, &mut IgnoreDiagnostics),
        }
    }
}

impl Width {
    fn len(self) -> i32 {
        match self {
            Width::Byte => 1,
            Width::Word => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::codebase::fake::MockFileSystem;
    use crate::diagnostics::{IgnoreDiagnostics, Message, MockSpan, TestDiagnosticsListener};
    use crate::expr::Expr;
    use crate::expr::*;
    use crate::span::fake::FakeSpanSystem;
    use crate::span::WithSpan;

    use std::marker::PhantomData;

    struct FakeMetadata<S>(PhantomData<S>);

    impl<S> FakeMetadata<S> {
        fn new() -> Self {
            Self(PhantomData)
        }
    }

    impl<S: Clone> SpanSource for FakeMetadata<S> {
        type Span = S;
    }

    impl<'a, D, S: Clone> ImportMetadata<FakeMetadata<S>> for Session<'a, D, FakeSpanSystem<S>> {
        type SpanPatcher = FakeSpanPatcher;

        fn import_metadata(&mut self, _: FakeMetadata<S>) -> Self::SpanPatcher {
            FakeSpanPatcher
        }
    }

    struct FakeSpanPatcher;

    impl<S> PatchSpan<S> for FakeSpanPatcher {
        fn patch_span(&self, _: &mut S) {}
    }

    #[test]
    fn section_with_immediate_byte_fragment() {
        let object = ObjectData {
            content: Content {
                sections: vec![Section {
                    constraints: Constraints { addr: None },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![Fragment::Immediate(
                        Expr::from_atom(Atom::Const(0xff), ()),
                        Width::Byte,
                    )],
                }],
                symbols: SymbolTable(vec![]),
            },
            metadata: FakeMetadata::new(),
        };
        let program = {
            let mut fs = MockFileSystem::new();
            let mut session = Session::<_, FakeSpanSystem<_>>::new(&mut fs, IgnoreDiagnostics);
            session.import_object(object);
            session.link()
        };
        assert_eq!(*program.sections[0].data, [0xff])
    }

    #[test]
    fn section_with_two_immediate_byte_fragments() {
        let object = ObjectData {
            content: Content {
                sections: vec![Section {
                    constraints: Constraints { addr: None },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![
                        Fragment::Immediate(Expr::from_atom(Atom::Const(0x12), ()), Width::Byte),
                        Fragment::Immediate(Expr::from_atom(Atom::Const(0x34), ()), Width::Byte),
                    ],
                }],
                symbols: SymbolTable(vec![]),
            },
            metadata: FakeMetadata::new(),
        };
        let program = {
            let mut fs = MockFileSystem::new();
            let mut session = Session::<_, FakeSpanSystem<_>>::new(&mut fs, IgnoreDiagnostics);
            session.import_object(object);
            session.link()
        };
        assert_eq!(*program.sections[0].data, [0x12, 0x34])
    }

    #[test]
    fn diagnose_immediate_byte_fragment_less_than_negative_128() {
        test_diagnostic_for_out_of_range_immediate_byte_fragment(-129)
    }

    #[test]
    fn diagnose_immediate_byte_fragment_greater_than_255() {
        test_diagnostic_for_out_of_range_immediate_byte_fragment(256)
    }

    fn test_diagnostic_for_out_of_range_immediate_byte_fragment(value: i32) {
        let object = ObjectData {
            content: Content {
                sections: vec![Section {
                    constraints: Constraints { addr: None },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![Fragment::Immediate(
                        Expr::from_atom(Atom::Const(value), MockSpan::from("byte")),
                        Width::Byte,
                    )],
                }],
                symbols: SymbolTable(vec![]),
            },
            metadata: FakeMetadata::new(),
        };
        let listener = TestDiagnosticsListener::new();
        let diagnostics = listener.diagnostics.clone();
        let mut fs = MockFileSystem::new();
        let mut session = Session::<_, FakeSpanSystem<_>>::new(&mut fs, listener);
        session.import_object(object);
        session.link();
        assert_eq!(
            *diagnostics.into_inner(),
            [Message::ValueOutOfRange {
                value,
                width: Width::Byte,
            }
            .at(MockSpan::from("byte"))
            .into()]
        )
    }

    #[test]
    fn diagnose_unresolved_symbol() {
        // DW   name
        let object = ObjectData {
            content: Content {
                sections: vec![Section {
                    constraints: Constraints { addr: None },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![Fragment::Immediate(
                        Expr::from_atom(
                            Atom::Name(Symbol::UserDef(UserDefId(0))),
                            MockSpan::from("name"),
                        ),
                        Width::Word,
                    )],
                }],
                symbols: SymbolTable(vec![None]),
            },
            metadata: FakeMetadata::new(),
        };
        let listener = TestDiagnosticsListener::new();
        let diagnostics = listener.diagnostics.clone();
        let mut fs = MockFileSystem::new();
        let mut session = Session::<_, FakeSpanSystem<_>>::new(&mut fs, listener);
        session.import_object(object);
        session.link();
        assert_eq!(
            *diagnostics.into_inner(),
            [Message::UnresolvedSymbol {
                symbol: MockSpan::from("name")
            }
            .at(MockSpan::from("name"))
            .into()]
        )
    }

    #[test]
    fn diagnose_two_unresolved_symbols_in_one_expr() {
        // DW   name1 - name2
        let object = ObjectData {
            content: Content {
                sections: vec![Section {
                    constraints: Constraints { addr: None },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![Fragment::Immediate(
                        Expr(vec![
                            ExprOp::Atom(Atom::Name(Symbol::UserDef(UserDefId(0))))
                                .with_span(MockSpan::from("name1")),
                            ExprOp::Atom(Atom::Name(Symbol::UserDef(UserDefId(1))))
                                .with_span(MockSpan::from("name2")),
                            ExprOp::Binary(BinOp::Minus).with_span(MockSpan::from("diff")),
                        ]),
                        Width::Word,
                    )],
                }],
                symbols: SymbolTable(vec![None, None]),
            },
            metadata: FakeMetadata::new(),
        };
        let listener = TestDiagnosticsListener::new();
        let diagnostics = listener.diagnostics.clone();
        let mut fs = MockFileSystem::new();
        let mut session = Session::<_, FakeSpanSystem<_>>::new(&mut fs, listener);
        session.import_object(object);
        session.link();
        assert_eq!(
            *diagnostics.into_inner(),
            [
                Message::UnresolvedSymbol {
                    symbol: MockSpan::from("name1")
                }
                .at(MockSpan::from("name1"))
                .into(),
                Message::UnresolvedSymbol {
                    symbol: MockSpan::from("name2")
                }
                .at(MockSpan::from("name2"))
                .into()
            ]
        )
    }

    #[test]
    fn emit_symbol_after_definition() {
        // name DB  name
        let object = ObjectData {
            content: Content {
                sections: vec![Section {
                    constraints: Constraints { addr: None },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![
                        Fragment::Reloc(VarId(2)),
                        Fragment::Immediate(
                            Expr::from_atom(Atom::Name(Symbol::UserDef(UserDefId(0))), ()),
                            Width::Word,
                        ),
                    ],
                }],
                symbols: SymbolTable(vec![Some(UserDef::Closure(Closure {
                    expr: Expr::from_atom(Atom::Location, ()),
                    location: VarId(2),
                }))]),
            },
            metadata: FakeMetadata::new(),
        };
        let program = {
            let mut fs = MockFileSystem::new();
            let mut session = Session::<_, FakeSpanSystem<_>>::new(&mut fs, IgnoreDiagnostics);
            session.import_object(object);
            session.link()
        };
        assert_eq!(*program.sections[0].data, [0x00, 0x00])
    }

    #[test]
    fn emit_symbol_before_definition() {
        // name DB  name
        let object = ObjectData {
            content: Content {
                sections: vec![Section {
                    constraints: Constraints { addr: None },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![
                        Fragment::Immediate(
                            Expr::from_atom(Atom::Name(Symbol::UserDef(UserDefId(0))), ()),
                            Width::Word,
                        ),
                        Fragment::Reloc(VarId(2)),
                    ],
                }],
                symbols: SymbolTable(vec![Some(UserDef::Closure(Closure {
                    expr: Expr::from_atom(Atom::Location, ()),
                    location: VarId(2),
                }))]),
            },
            metadata: FakeMetadata::new(),
        };
        let program = {
            let mut fs = MockFileSystem::new();
            let mut session = Session::<_, FakeSpanSystem<_>>::new(&mut fs, IgnoreDiagnostics);
            session.import_object(object);
            session.link()
        };
        assert_eq!(*program.sections[0].data, [0x02, 0x00])
    }

    #[test]
    fn resolve_origin_relative_to_previous_section() {
        let origin1 = 0x150;
        let skipped_bytes = 0x10;

        // ORG $0150
        // NOP
        // ORG . + $10
        // HALT
        let object = ObjectData {
            content: Content {
                sections: vec![
                    Section {
                        constraints: Constraints {
                            addr: Some(Expr::from_atom(Atom::Const(0x0150), ())),
                        },
                        addr: VarId(0),
                        size: VarId(1),
                        fragments: vec![Fragment::Byte(0x00)],
                    },
                    Section {
                        constraints: Constraints {
                            addr: Some(Expr(vec![
                                ExprOp::Atom(Atom::Location).with_span(()),
                                ExprOp::Atom(Atom::Const(0x10)).with_span(()),
                                ExprOp::Binary(BinOp::Plus).with_span(()),
                            ])),
                        },
                        addr: VarId(2),
                        size: VarId(3),
                        fragments: vec![Fragment::Byte(0x76)],
                    },
                ],
                symbols: SymbolTable(vec![]),
            },
            metadata: FakeMetadata::new(),
        };

        let program = {
            let mut fs = MockFileSystem::new();
            let mut session = Session::<_, FakeSpanSystem<_>>::new(&mut fs, IgnoreDiagnostics);
            session.import_object(object);
            session.link()
        };
        assert_eq!(
            program.sections[1].addr,
            (origin1 + 1 + skipped_bytes) as usize
        )
    }

    #[test]
    fn label_defined_as_section_origin_plus_offset() {
        let addr = 0xffe1;

        // ORG $ffe1
        // LABEL
        let content = Content {
            sections: vec![Section {
                constraints: Constraints {
                    addr: Some(Expr::from_atom(Atom::Const(addr), ())),
                },
                addr: VarId(0),
                size: VarId(1),
                fragments: vec![Fragment::Reloc(VarId(2))],
            }],
            symbols: SymbolTable(vec![Some(UserDef::Closure(Closure {
                expr: Expr::from_atom(Atom::Location, ()),
                location: VarId(2),
            }))]),
        };
        let mut vars = VarTable(vec![addr.into(), 0.into(), addr.into()]);

        vars.resolve(&content);
        assert_eq!(vars[VarId(0)], addr.into());
    }

    #[test]
    fn empty_section_has_size_zero() {
        assert_section_size(
            0,
            Content {
                sections: vec![Section {
                    constraints: Constraints { addr: None },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![],
                }],
                symbols: SymbolTable::new(),
            },
            VarTable(vec![0x0000.into(), 0.into()]),
        )
    }

    #[test]
    fn section_with_one_byte_has_size_one() {
        assert_section_size(
            1,
            Content {
                sections: vec![Section {
                    constraints: Constraints { addr: None },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![Fragment::Byte(0x00)],
                }],
                symbols: SymbolTable::new(),
            },
            VarTable(vec![0x0000.into(), 1.into()]),
        );
    }

    #[test]
    fn section_with_const_inline_addr_ld_has_size_two() {
        test_section_size_with_literal_ld_inline_addr(0xff00, 2)
    }

    #[test]
    fn section_with_const_inline_addr_ld_has_size_three() {
        test_section_size_with_literal_ld_inline_addr(0x1234, 3)
    }

    fn test_section_size_with_literal_ld_inline_addr(addr: i32, expected: i32) {
        assert_section_size(
            expected,
            Content {
                sections: vec![Section {
                    constraints: Constraints { addr: None },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![Fragment::LdInlineAddr(0xf0, addr.into())],
                }],
                symbols: SymbolTable::new(),
            },
            VarTable(vec![0x0000.into(), Var::Unknown]),
        );
    }

    #[test]
    fn ld_inline_addr_with_symbol_after_instruction_has_size_three() {
        assert_section_size(
            3,
            Content {
                sections: vec![Section {
                    constraints: Constraints { addr: None },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![
                        Fragment::LdInlineAddr(
                            0xf0,
                            Atom::Name(Symbol::UserDef(UserDefId(0))).into(),
                        ),
                        Fragment::Reloc(VarId(2)),
                    ],
                }],
                symbols: SymbolTable(vec![Some(UserDef::Closure(Closure {
                    expr: Expr::from_atom(Atom::Location, ()),
                    location: VarId(2),
                }))]),
            },
            VarTable(vec![
                0x0000.into(),
                Var::Range { min: 2, max: 3 },
                Var::Range { min: 2, max: 3 },
            ]),
        )
    }

    #[test]
    fn resolve_expr_with_section_addr() {
        // my_section   SECTION
        //              ORG     $1337
        //              DW      my_section
        let object = ObjectData {
            content: Content {
                sections: vec![Section {
                    constraints: Constraints {
                        addr: Some(Expr::from_atom(Atom::Const(0x1337), ())),
                    },
                    addr: VarId(0),
                    size: VarId(1),
                    fragments: vec![Fragment::Immediate(
                        Expr::from_atom(Atom::Name(Symbol::UserDef(UserDefId(0))), ()),
                        Width::Word,
                    )],
                }],
                symbols: SymbolTable(vec![Some(UserDef::Section(SectionId(0)))]),
            },
            metadata: FakeMetadata::new(),
        };

        let program = {
            let mut fs = MockFileSystem::new();
            let mut session = Session::<_, FakeSpanSystem<_>>::new(&mut fs, IgnoreDiagnostics);
            session.import_object(object);
            session.link()
        };
        assert_eq!(*program.sections[0].data, [0x37, 0x13])
    }

    #[test]
    fn traverse_reserved_bytes() {
        let addr = 0x0100;
        let bytes = 10;
        let symbol = VarId(2);

        //          ORG $0100
        //          DS  10
        // label    DW  label
        let content = Content {
            sections: vec![Section {
                constraints: Constraints {
                    addr: Some(Expr::from_atom(Atom::Const(addr), ())),
                },
                addr: VarId(0),
                size: VarId(1),
                fragments: vec![
                    Fragment::Reserved(Expr::from_atom(Atom::Const(bytes), ())),
                    Fragment::Reloc(VarId(2)),
                    Fragment::Immediate(
                        Expr::from_atom(Atom::Name(Symbol::UserDef(UserDefId(0))), ()),
                        Width::Word,
                    ),
                ],
            }],
            symbols: SymbolTable(vec![Some(UserDef::Closure(Closure {
                expr: Expr::from_atom(Atom::Name(Symbol::UserDef(UserDefId(0))), ()),
                location: VarId(2),
            }))]),
        };
        let mut vars = VarTable(vec![addr.into(), (bytes + 2).into(), (addr + bytes).into()]);

        vars.resolve(&content);
        assert_eq!(vars[symbol], (addr + bytes).into())
    }

    fn assert_section_size(expected: impl Into<Var>, content: Content<()>, mut vars: VarTable) {
        vars.resolve(&content);
        assert_eq!(
            vars[content.sections().next().unwrap().size],
            expected.into()
        );
    }
}
