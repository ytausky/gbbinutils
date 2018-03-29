use frontend::{Keyword, OperationReceiver, StrExprFactory};
use frontend::syntax::{self, StrToken, SynExpr};

use std::marker::PhantomData;

mod instruction;

pub struct SemanticActions<'actions, 'session, 'src, OR>
where
    'session: 'actions,
    'src: 'actions,
    OR: 'session + OperationReceiver<'src>,
{
    session: &'session mut OR,
    contexts: Vec<Context<'src>>,
    _phantom: PhantomData<&'actions ()>,
}

enum Context<'a> {
    Block,
    Instruction(syntax::StrToken<'a>, Vec<SynExpr<syntax::StrToken<'a>>>),
}

impl<'actions, 'session, 'src, OR> SemanticActions<'actions, 'session, 'src, OR>
where
    'session: 'actions,
    'src: 'actions,
    OR: 'session + OperationReceiver<'src>,
{
    pub fn new(session: &'session mut OR) -> SemanticActions<'actions, 'session, 'src, OR> {
        SemanticActions {
            session,
            contexts: vec![Context::Block],
            _phantom: PhantomData,
        }
    }
}

impl<'actions, 'session, 'src, OR> syntax::BlockContext
    for SemanticActions<'actions, 'session, 'src, OR>
where
    'session: 'actions,
    'src: 'actions,
    OR: 'session + OperationReceiver<'src>,
{
    type Terminal = StrToken<'src>;
    type CommandContext = Self;
    type TerminalSequenceContext = Self;

    fn add_label(&mut self, label: Self::Terminal) {
        match label {
            StrToken::Label(spelling) => self.session.define_label(spelling),
            _ => panic!(),
        }
    }

    fn enter_command(&mut self, name: Self::Terminal) -> &mut Self::CommandContext {
        self.contexts.push(Context::Instruction(name, vec![]));
        self
    }

    fn enter_macro_definition(
        &mut self,
        _label: Self::Terminal,
    ) -> &mut Self::TerminalSequenceContext {
        unimplemented!()
    }
}

impl<'actions, 'session, 'src, OR> syntax::CommandContext
    for SemanticActions<'actions, 'session, 'src, OR>
where
    'session: 'actions,
    'src: 'actions,
    OR: 'session + OperationReceiver<'src>,
{
    type Terminal = StrToken<'src>;

    fn add_argument(&mut self, expr: SynExpr<Self::Terminal>) {
        match self.contexts.last_mut() {
            Some(&mut Context::Instruction(_, ref mut args)) => args.push(expr),
            _ => panic!(),
        }
    }

    fn exit_command(&mut self) {
        if let Some(Context::Instruction(name, args)) = self.contexts.pop() {
            match name {
                StrToken::Keyword(Keyword::Include) => {
                    self.session.include_source_file(reduce_include(args))
                }
                StrToken::Keyword(keyword) => {
                    let mut analyzer =
                        self::instruction::CommandAnalyzer::new(StrExprFactory::new());
                    self.session.emit_instruction(
                        analyzer
                            .analyze_instruction(keyword, args.into_iter())
                            .unwrap(),
                    )
                }
                StrToken::Identifier(ident) => {
                    println!("Probably macro invocation: {:?} {:?}", ident, args)
                }
                _ => panic!(),
            }
        } else {
            panic!()
        }
    }
}

impl<'actions, 'session, 'src, OR> syntax::TerminalSequenceContext
    for SemanticActions<'actions, 'session, 'src, OR>
where
    'session: 'actions,
    'src: 'actions,
    OR: 'session + OperationReceiver<'src>,
{
    type Terminal = StrToken<'src>;

    fn push_terminal(&mut self, _terminal: Self::Terminal) {
        unimplemented!()
    }

    fn exit_terminal_sequence(&mut self) {
        unimplemented!()
    }
}

fn reduce_include(mut arguments: Vec<SynExpr<StrToken>>) -> &str {
    assert_eq!(arguments.len(), 1);
    let path = arguments.pop().unwrap();
    match path {
        SynExpr::Atom(StrToken::QuotedString(path_str)) => path_str,
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use frontend::syntax::{BlockContext, CommandContext};
    use ir;

    struct TestOperationReceiver(Vec<TestOperation>);

    impl TestOperationReceiver {
        fn new() -> TestOperationReceiver {
            TestOperationReceiver(Vec::new())
        }
    }

    #[derive(Debug, PartialEq)]
    enum TestOperation {
        Include(&'static str),
        Instruction(ir::Instruction),
        Label(&'static str),
    }

    impl OperationReceiver<'static> for TestOperationReceiver {
        fn include_source_file(&mut self, filename: &'static str) {
            self.0.push(TestOperation::Include(filename))
        }

        fn emit_instruction(&mut self, instruction: ir::Instruction) {
            self.0.push(TestOperation::Instruction(instruction))
        }

        fn define_label(&mut self, label: &'static str) {
            self.0.push(TestOperation::Label(label))
        }
    }

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let actions = collect_semantic_actions(|mut actions| {
            actions.enter_command(StrToken::Keyword(Keyword::Include));
            let expr = SynExpr::from(StrToken::QuotedString(filename));
            actions.add_argument(expr);
            actions.exit_command();
        });
        assert_eq!(actions, [TestOperation::Include(filename)])
    }

    #[test]
    fn analyze_label() {
        let actions =
            collect_semantic_actions(|mut actions| actions.add_label(StrToken::Label("label")));
        assert_eq!(actions, [TestOperation::Label("label")])
    }

    fn collect_semantic_actions<F>(f: F) -> Vec<TestOperation>
    where
        F: for<'actions, 'session: 'actions> FnOnce(
            SemanticActions<
                'actions,
                'session,
                'static,
                TestOperationReceiver,
            >,
        ),
    {
        let mut operations = TestOperationReceiver::new();
        f(SemanticActions::new(&mut operations));
        operations.0
    }
}
