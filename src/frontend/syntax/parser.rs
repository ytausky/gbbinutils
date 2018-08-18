use super::*;
use diagnostics::{Diagnostic, Message, SourceRange};

use std::iter;

#[derive(Clone, Debug, PartialEq)]
pub enum Token<S: TokenSpec> {
    ClosingParenthesis,
    Colon,
    Comma,
    Command(S::Command),
    Endm,
    Eol,
    Ident(S::Ident),
    Literal(S::Literal),
    Macro,
    OpeningParenthesis,
}

impl Copy for Token<()> {}

impl<S: TokenSpec> Token<S> {
    fn kind(&self) -> Token<()> {
        use self::Token::*;
        match *self {
            ClosingParenthesis => ClosingParenthesis,
            Colon => Colon,
            Comma => Comma,
            Command(_) => Command(()),
            Endm => Endm,
            Eol => Eol,
            Ident(_) => Ident(()),
            Literal(_) => Literal(()),
            Macro => Macro,
            OpeningParenthesis => OpeningParenthesis,
        }
    }
}

type Lookahead = Option<Token<()>>;

fn follows_line(lookahead: Lookahead) -> bool {
    match lookahead {
        None | Some(Token::Eol) => true,
        _ => false,
    }
}

pub fn parse_src<S: TokenSpec, T: SourceRange, I, F>(tokens: I, actions: F)
where
    I: Iterator<Item = (Token<S>, T)>,
    F: FileContext<S, T>,
{
    let mut parser = Parser {
        tokens: tokens.peekable(),
    };
    parser.parse_file(actions)
}

struct Parser<I: Iterator> {
    tokens: iter::Peekable<I>,
}

macro_rules! mk_expect {
    ($name:ident, $ret_ty:ident) => {
        fn $name(&mut self) -> (S::$ret_ty, T) {
            match self.tokens.next() {
                Some((Token::$ret_ty(inner), t)) => (inner, t),
                _ => panic!(),
            }
        }
    }
}

impl<S: TokenSpec, T: SourceRange, I: Iterator<Item = (Token<S>, T)>> Parser<I> {
    mk_expect!(expect_command, Command);
    mk_expect!(expect_ident, Ident);

    fn lookahead(&mut self) -> Lookahead {
        self.tokens.peek().map(|&(ref t, _)| t.kind())
    }

    fn bump(&mut self) -> I::Item {
        self.tokens.next().unwrap()
    }

    fn expect(&mut self, expected: Lookahead) -> I::Item {
        assert_eq!(self.lookahead(), expected);
        self.bump()
    }

    fn parse_file<F: FileContext<S, T>>(&mut self, actions: F) {
        self.parse_terminated_list(
            Some(Token::Eol),
            |l| l.is_none(),
            |p, c| p.parse_line(c),
            actions,
        );
    }

    fn parse_line<F: FileContext<S, T>>(&mut self, actions: F) -> F {
        if self.lookahead() == Some(Token::Ident(())) {
            self.parse_potentially_labeled_line(actions)
        } else {
            self.parse_unlabeled_line(actions.enter_line(None)).exit()
        }
    }

    fn parse_potentially_labeled_line<F: FileContext<S, T>>(&mut self, actions: F) -> F {
        let ident = self.expect_ident();
        if self.lookahead() == Some(Token::Colon) {
            self.bump();
            self.parse_unlabeled_line(actions.enter_line(Some(ident)))
        } else {
            self.parse_macro_invocation(ident, actions.enter_line(None))
        }.exit()
    }

    fn parse_unlabeled_line<LA: LineActions<S, T>>(&mut self, actions: LA) -> LA {
        match self.lookahead() {
            None | Some(Token::Eol) => actions,
            Some(Token::Command(())) => self.parse_command(actions),
            Some(Token::Ident(())) => {
                let ident = self.expect_ident();
                self.parse_macro_invocation(ident, actions)
            }
            Some(Token::Macro) => self.parse_macro_def(actions),
            _ => {
                let (_, range) = self.bump();
                actions.emit_diagnostic(Diagnostic::new(
                    Message::UnexpectedToken {
                        token: range.clone(),
                    },
                    range,
                ));
                actions
            }
        }
    }

    fn parse_macro_def<LA: LineActions<S, T>>(&mut self, actions: LA) -> LA {
        self.expect(Some(Token::Macro));
        let mut macro_body_actions = self.parse_terminated_list(
            Some(Token::Comma),
            follows_line,
            |p, a| p.parse_macro_param(a),
            actions.enter_macro_def(),
        ).exit();
        self.expect(Some(Token::Eol));
        while self.lookahead() != Some(Token::Endm) {
            macro_body_actions.push_token(self.bump())
        }
        self.expect(Some(Token::Endm));
        macro_body_actions.exit()
    }

    fn parse_macro_param<MPA>(&mut self, mut actions: MPA) -> MPA
    where
        MPA: MacroParamsActions<T, TokenSpec = S>,
    {
        actions.add_parameter(self.expect_ident());
        actions
    }

    fn parse_command<LA: LineActions<S, T>>(&mut self, actions: LA) -> LA {
        let first_token = self.expect_command();
        let mut command_context = actions.enter_command(first_token);
        command_context = self.parse_argument_list(command_context);
        command_context.exit()
    }

    fn parse_macro_invocation<LA: LineActions<S, T>>(
        &mut self,
        name: (S::Ident, T),
        actions: LA,
    ) -> LA {
        let mut invocation_context = actions.enter_macro_invocation(name);
        invocation_context = self.parse_macro_arg_list(invocation_context);
        invocation_context.exit()
    }

    fn parse_argument_list<C: CommandContext<T, TokenSpec = S>>(&mut self, actions: C) -> C {
        self.parse_terminated_list(
            Some(Token::Comma),
            follows_line,
            |p, c| p.parse_argument(c),
            actions,
        )
    }

    fn parse_macro_arg_list<M: MacroInvocationContext<T, Token = Token<S>>>(
        &mut self,
        actions: M,
    ) -> M {
        self.parse_terminated_list(
            Some(Token::Comma),
            follows_line,
            |p, actions| {
                let mut arg_context = actions.enter_macro_arg();
                let mut next_token = p.lookahead();
                while next_token != Some(Token::Comma) && !follows_line(next_token) {
                    arg_context.push_token(p.bump());
                    next_token = p.lookahead()
                }
                arg_context.exit()
            },
            actions,
        )
    }

    fn parse_terminated_list<FP, P, C>(
        &mut self,
        delimiter: Lookahead,
        mut follow: FP,
        parser: P,
        mut context: C,
    ) -> C
    where
        FP: FnMut(Lookahead) -> bool,
        P: FnMut(&mut Self, C) -> C,
        C: DiagnosticsListener<T>,
    {
        context = self.parse_list(delimiter, &mut follow, parser, context);
        if !follow(self.lookahead()) {
            let unexpected_range = self.tokens.peek().unwrap().1.clone();
            context.emit_diagnostic(Diagnostic::new(
                Message::UnexpectedToken {
                    token: unexpected_range.clone(),
                },
                unexpected_range,
            ));
            while !follow(self.lookahead()) {
                self.bump();
            }
        }
        context
    }

    fn parse_list<FP, P, C>(
        &mut self,
        delimiter: Lookahead,
        follow: &mut FP,
        mut parser: P,
        context: C,
    ) -> C
    where
        FP: FnMut(Lookahead) -> bool,
        P: FnMut(&mut Self, C) -> C,
        C: DiagnosticsListener<T>,
    {
        if follow(self.lookahead()) {
            context
        } else {
            self.parse_nonempty_list(delimiter, &mut parser, context)
        }
    }

    fn parse_nonempty_list<P, C>(
        &mut self,
        delimiter: Lookahead,
        parser: &mut P,
        mut actions: C,
    ) -> C
    where
        P: FnMut(&mut Self, C) -> C,
        C: DiagnosticsListener<T>,
    {
        actions = parser(self, actions);
        while self.lookahead() == delimiter {
            self.bump();
            actions = parser(self, actions)
        }
        actions
    }

    fn parse_argument<C: CommandContext<T, TokenSpec = S>>(&mut self, actions: C) -> C {
        self.parse_expression(actions.add_argument()).exit()
    }

    fn parse_expression<EA: ExprActions<T, TokenSpec = S>>(&mut self, actions: EA) -> EA {
        if self.lookahead() == Some(Token::OpeningParenthesis) {
            self.parse_parenthesized_expression(actions)
        } else {
            self.parse_atomic_expr(actions)
        }
    }

    fn parse_parenthesized_expression<EA: ExprActions<T, TokenSpec = S>>(
        &mut self,
        actions: EA,
    ) -> EA {
        let (_, left) = self.expect(Some(Token::OpeningParenthesis));
        let mut actions = self.parse_expression(actions);
        let (_, right) = self.expect(Some(Token::ClosingParenthesis));
        actions.apply_operator((ExprOperator::Parentheses, left.extend(&right)));
        actions
    }

    fn parse_atomic_expr<EA: ExprActions<T, TokenSpec = S>>(&mut self, mut actions: EA) -> EA {
        let (token, interval) = self.bump();
        actions.push_atom((
            match token {
                Token::Ident(ident) => ExprAtom::Ident(ident),
                Token::Literal(literal) => ExprAtom::Literal(literal),
                _ => panic!(),
            },
            interval,
        ));
        actions
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_src, Token::{self, *},
    };

    use diagnostics::{Diagnostic, DiagnosticsListener, Message, SourceRange};
    use frontend::syntax::{self, ExprAtom, ExprOperator, TokenSpec};
    use std::borrow::Borrow;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::fmt::Debug;

    #[derive(Clone, Debug, PartialEq)]
    struct Symbolic;

    impl TokenSpec for Symbolic {
        type Command = SymCommand;
        type Ident = SymIdent;
        type Literal = SymLiteral;
    }

    #[derive(Clone, Debug, PartialEq)]
    struct SymCommand(usize);

    #[derive(Clone, Debug, PartialEq)]
    struct SymIdent(usize);

    #[derive(Clone, Debug, PartialEq)]
    struct SymLiteral(usize);

    type SymToken = Token<Symbolic>;

    fn mk_sym_token(id: usize, token: Token<()>) -> SymToken {
        match token {
            Command(()) => Command(SymCommand(id)),
            Ident(()) => Ident(SymIdent(id)),
            Literal(()) => Literal(SymLiteral(id)),
            ClosingParenthesis => ClosingParenthesis,
            Colon => Colon,
            Comma => Comma,
            Endm => Endm,
            Eol => Eol,
            Macro => Macro,
            OpeningParenthesis => OpeningParenthesis,
        }
    }

    struct InputTokens {
        tokens: Vec<SymToken>,
        names: HashMap<String, usize>,
    }

    macro_rules! add_token {
        ($input:expr, $token:expr) => {
            let id = $input.tokens.len();
            $input.tokens.push(mk_sym_token(id, $token))
        };
        ($input:expr, $name:ident @ $token:expr) => {
            $input
                .names
                .insert(stringify!($name).into(), $input.tokens.len());
            add_token!($input, $token)
        };
    }

    macro_rules! input_tokens_impl {
        ($input:expr, ) => {};
        ($input:expr, $token:expr) => {
            add_token!($input, $token)
        };
        ($input:expr, $token:expr, $($tail:tt)*) => {
            add_token!($input, $token);
            input_tokens_impl![$input, $($tail)*]
        };
        ($input:expr, $name:ident @ $token:expr) => {
            add_token!($input, $name @ $token)
        };
        ($input:expr, $name:ident @ $token:expr, $($tail:tt)*) => {
            add_token!($input, $name @ $token);
            input_tokens_impl![$input, $($tail)*]
        }
    }

    macro_rules! input_tokens {
        () => {
            InputTokens {
                tokens: Vec::new(),
                names: HashMap::new(),
            }
        };
        ($($tokens:tt)*) => {{
            let mut input = input_tokens![];
            input_tokens_impl!(input, $($tokens)*);
            input
        }};
    }

    #[test]
    fn test_token_macro() {
        let tokens = input_tokens![
            my_tok @ Command(()),
            Literal(()),
            next_one @ Macro,
        ];
        assert_eq!(
            tokens.tokens,
            [Command(SymCommand(0)), Literal(SymLiteral(1)), Macro]
        );
        assert_eq!(tokens.names.get("my_tok"), Some(&0));
        assert_eq!(tokens.names.get("next_one"), Some(&2))
    }

    #[derive(Clone, Debug, PartialEq)]
    struct SymRange<T> {
        start: T,
        end: T,
    }

    impl<T: Clone> From<T> for SymRange<T> {
        fn from(x: T) -> SymRange<T> {
            SymRange {
                start: x.clone(),
                end: x,
            }
        }
    }

    impl<T: Clone + Debug> SourceRange for SymRange<T> {
        fn extend(&self, other: &Self) -> Self {
            SymRange {
                start: self.start.clone(),
                end: other.end.clone(),
            }
        }
    }

    #[test]
    fn parse_empty_src() {
        assert_eq_actions(input_tokens![], file([]))
    }

    struct TestContext {
        actions: RefCell<Vec<Action>>,
        token_seq_kind: Option<TokenSeqKind>,
    }

    impl TestContext {
        fn new() -> TestContext {
            TestContext {
                actions: RefCell::new(Vec::new()),
                token_seq_kind: None,
            }
        }
    }

    #[derive(Debug, PartialEq)]
    enum Action {
        AddParameter(SymIdent),
        ApplyExprOperator(ExprOperator),
        EnterArgument,
        EnterInstruction(SymCommand),
        EnterLine(Option<SymIdent>),
        EnterMacroArg,
        EnterMacroBody,
        EnterMacroDef,
        EnterMacroInvocation(SymIdent),
        Error(Diagnostic<SymRange<usize>>),
        ExitArgument,
        ExitInstruction,
        ExitLine,
        ExitMacroArg,
        ExitMacroDef,
        ExitMacroInvocation,
        PushExprAtom(ExprAtom<Symbolic>),
        PushTerminal(usize),
    }

    enum TokenSeqKind {
        MacroArg,
        MacroDef,
    }

    impl<'a> DiagnosticsListener<SymRange<usize>> for &'a mut TestContext {
        fn emit_diagnostic(&self, diagnostic: Diagnostic<SymRange<usize>>) {
            self.actions.borrow_mut().push(Action::Error(diagnostic))
        }
    }

    impl<'a> syntax::FileContext<Symbolic, SymRange<usize>> for &'a mut TestContext {
        type LineActions = Self;

        fn enter_line(self, label: Option<(SymIdent, SymRange<usize>)>) -> Self::LineActions {
            self.actions
                .borrow_mut()
                .push(Action::EnterLine(label.map(|(ident, _)| ident)));
            self
        }
    }

    impl<'a> syntax::LineActions<Symbolic, SymRange<usize>> for &'a mut TestContext {
        type CommandContext = Self;
        type MacroParamsActions = Self;
        type MacroInvocationContext = Self;
        type Parent = Self;

        fn enter_command(
            self,
            (command, _): (SymCommand, SymRange<usize>),
        ) -> Self::CommandContext {
            self.actions
                .borrow_mut()
                .push(Action::EnterInstruction(command));
            self
        }

        fn enter_macro_def(self) -> Self::MacroParamsActions {
            self.actions.borrow_mut().push(Action::EnterMacroDef);
            self.token_seq_kind = Some(TokenSeqKind::MacroDef);
            self
        }

        fn enter_macro_invocation(
            self,
            name: (SymIdent, SymRange<usize>),
        ) -> Self::MacroInvocationContext {
            self.actions
                .borrow_mut()
                .push(Action::EnterMacroInvocation(name.0));
            self
        }

        fn exit(self) -> Self::Parent {
            self.actions.borrow_mut().push(Action::ExitLine);
            self
        }
    }

    impl<'a> syntax::CommandContext<SymRange<usize>> for &'a mut TestContext {
        type TokenSpec = Symbolic;
        type ArgActions = Self;
        type Parent = Self;

        fn add_argument(self) -> Self::ArgActions {
            self.actions.borrow_mut().push(Action::EnterArgument);
            self
        }

        fn exit(self) -> Self::Parent {
            self.actions.borrow_mut().push(Action::ExitInstruction);
            self
        }
    }

    impl<'a> syntax::ExprActions<SymRange<usize>> for &'a mut TestContext {
        type TokenSpec = Symbolic;
        type Parent = Self;

        fn push_atom(&mut self, atom: (ExprAtom<Symbolic>, SymRange<usize>)) {
            self.actions.borrow_mut().push(Action::PushExprAtom(atom.0))
        }

        fn apply_operator(&mut self, operator: (ExprOperator, SymRange<usize>)) {
            self.actions
                .borrow_mut()
                .push(Action::ApplyExprOperator(operator.0))
        }

        fn exit(self) -> Self::Parent {
            self.actions.borrow_mut().push(Action::ExitArgument);
            self
        }
    }

    impl<'a> syntax::MacroParamsActions<SymRange<usize>> for &'a mut TestContext {
        type TokenSpec = Symbolic;
        type MacroBodyActions = Self;
        type Parent = Self;

        fn add_parameter(&mut self, (ident, _): (SymIdent, SymRange<usize>)) {
            self.actions.borrow_mut().push(Action::AddParameter(ident))
        }

        fn exit(self) -> Self::MacroBodyActions {
            self.actions.borrow_mut().push(Action::EnterMacroBody);
            self
        }
    }

    impl<'a> syntax::MacroInvocationContext<SymRange<usize>> for &'a mut TestContext {
        type Token = SymToken;
        type Parent = Self;
        type MacroArgContext = Self;

        fn enter_macro_arg(self) -> Self::MacroArgContext {
            self.actions.borrow_mut().push(Action::EnterMacroArg);
            self.token_seq_kind = Some(TokenSeqKind::MacroArg);
            self
        }

        fn exit(self) -> Self::Parent {
            self.actions.borrow_mut().push(Action::ExitMacroInvocation);
            self
        }
    }

    impl<'a> syntax::TokenSeqContext<SymRange<usize>> for &'a mut TestContext {
        type Token = SymToken;
        type Parent = Self;

        fn push_token(&mut self, token: (Self::Token, SymRange<usize>)) {
            let id = token.1.start;
            assert_eq!(id, token.1.end);
            self.actions.borrow_mut().push(Action::PushTerminal(id))
        }

        fn exit(self) -> Self::Parent {
            self.actions
                .borrow_mut()
                .push(match *self.token_seq_kind.as_ref().unwrap() {
                    TokenSeqKind::MacroArg => Action::ExitMacroArg,
                    TokenSeqKind::MacroDef => Action::ExitMacroDef,
                });
            self.token_seq_kind = None;
            self
        }
    }

    #[test]
    fn parse_empty_line() {
        assert_eq_actions(
            input_tokens![Eol],
            file([unlabeled(empty()), unlabeled(empty())]),
        )
    }

    fn assert_eq_actions(tokens: InputTokens, expected: File) {
        let mut parsing_context = TestContext::new();
        parse_src(
            tokens
                .tokens
                .iter()
                .cloned()
                .zip((0..).map(|n| SymRange::from(n))),
            &mut parsing_context,
        );
        assert_eq!(
            parsing_context.actions.into_inner(),
            expected.into_actions(&tokens)
        )
    }

    #[derive(Clone)]
    enum TokenRef {
        Id(usize),
        Name(String),
    }

    impl From<usize> for TokenRef {
        fn from(id: usize) -> Self {
            TokenRef::Id(id)
        }
    }

    impl From<&'static str> for TokenRef {
        fn from(name: &'static str) -> Self {
            TokenRef::Name(name.to_string())
        }
    }

    impl TokenRef {
        fn resolve(&self, input: &InputTokens) -> usize {
            match self {
                TokenRef::Id(id) => *id,
                TokenRef::Name(name) => *input.names.get(name).unwrap(),
            }
        }
    }

    impl SymRange<TokenRef> {
        fn resolve(&self, input: &InputTokens) -> SymRange<usize> {
            SymRange {
                start: self.start.resolve(input),
                end: self.end.resolve(input),
            }
        }
    }

    struct File(Vec<Line>);

    fn file(lines: impl Borrow<[Line]>) -> File {
        File(lines.borrow().iter().cloned().collect())
    }

    impl File {
        fn into_actions(self, input: &InputTokens) -> Vec<Action> {
            self.0
                .into_iter()
                .flat_map(|line| line.into_actions(input))
                .collect()
        }
    }

    #[test]
    fn parse_nullary_instruction() {
        assert_eq_actions(
            input_tokens![nop @ Command(())],
            file([unlabeled(command("nop", []))]),
        )
    }

    #[test]
    fn parse_nullary_instruction_after_eol() {
        assert_eq_actions(
            input_tokens![Eol, nop @ Command(())],
            file([unlabeled(empty()), unlabeled(command("nop", []))]),
        )
    }

    #[derive(Clone)]
    struct Line(Option<usize>, Option<LineBody>);

    fn labeled(label: usize, body: Option<LineBody>) -> Line {
        Line(Some(label), body)
    }

    fn unlabeled(body: Option<LineBody>) -> Line {
        Line(None, body)
    }

    impl Line {
        fn into_actions(self, input: &InputTokens) -> Vec<Action> {
            let mut actions = vec![Action::EnterLine(self.0.map(|id| SymIdent(id)))];
            if let Some(body) = self.1 {
                actions.append(&mut body.into_actions(input))
            }
            actions.push(Action::ExitLine);
            actions
        }
    }

    fn empty() -> Option<LineBody> {
        None
    }

    #[derive(Clone)]
    enum LineBody {
        Command(TokenRef, Vec<SymExpr>, Option<SymDiagnostic>),
        Error(SymDiagnostic),
        Invoke(usize, Vec<TokenSeq>),
        MacroDef(Vec<usize>, Vec<usize>),
    }

    #[derive(Clone)]
    struct SymDiagnostic {
        message_ctor: MessageCtor,
        ranges: Vec<SymRange<TokenRef>>,
        highlight: SymRange<TokenRef>,
    }

    type MessageCtor = fn(Vec<SymRange<usize>>) -> Message<SymRange<usize>>;

    impl From<SymDiagnostic> for LineBody {
        fn from(diagnostic: SymDiagnostic) -> Self {
            LineBody::Error(diagnostic)
        }
    }

    impl SymDiagnostic {
        fn into_action(self, input: &InputTokens) -> Action {
            let message = (self.message_ctor)(
                self.ranges
                    .into_iter()
                    .map(|range| range.resolve(input))
                    .collect(),
            );
            Action::Error(Diagnostic::new(message, self.highlight.resolve(input)))
        }
    }

    fn command(id: impl Into<TokenRef>, args: impl Borrow<[SymExpr]>) -> Option<LineBody> {
        Some(LineBody::Command(
            id.into(),
            args.borrow().iter().cloned().collect(),
            None,
        ))
    }

    fn malformed_command(
        id: impl Into<TokenRef>,
        args: impl Borrow<[SymExpr]>,
        diagnostic: SymDiagnostic,
    ) -> Option<LineBody> {
        Some(LineBody::Command(
            id.into(),
            args.borrow().iter().cloned().collect(),
            Some(diagnostic),
        ))
    }

    fn invoke(id: usize, args: impl Borrow<[TokenSeq]>) -> Option<LineBody> {
        Some(LineBody::Invoke(
            id,
            args.borrow().iter().cloned().collect(),
        ))
    }

    fn macro_def(params: impl Borrow<[usize]>, body: impl Borrow<[usize]>) -> Option<LineBody> {
        Some(LineBody::MacroDef(
            params.borrow().iter().cloned().collect(),
            body.borrow().iter().cloned().collect(),
        ))
    }

    impl LineBody {
        fn into_actions(self, input: &InputTokens) -> Vec<Action> {
            let mut actions = Vec::new();
            match self {
                LineBody::Command(id, args, error) => {
                    actions.push(Action::EnterInstruction(SymCommand(id.resolve(input))));
                    for mut arg in args {
                        actions.push(Action::EnterArgument);
                        actions.append(&mut arg.into_actions(input));
                        actions.push(Action::ExitArgument)
                    }
                    if let Some(diagnostic) = error {
                        actions.push(diagnostic.into_action(input))
                    }
                    actions.push(Action::ExitInstruction)
                }
                LineBody::Error(diagnostic) => actions.push(diagnostic.into_action(input)),
                LineBody::Invoke(id, args) => {
                    actions.push(Action::EnterMacroInvocation(SymIdent(id)));
                    for arg in args {
                        actions.push(Action::EnterMacroArg);
                        actions.append(&mut arg.into_actions());
                        actions.push(Action::ExitMacroArg)
                    }
                    actions.push(Action::ExitMacroInvocation)
                }
                LineBody::MacroDef(params, body) => {
                    actions.push(Action::EnterMacroDef);
                    actions.extend(
                        params
                            .into_iter()
                            .map(|id| Action::AddParameter(SymIdent(id))),
                    );
                    actions.push(Action::EnterMacroBody);
                    actions.extend(body.into_iter().map(|t| Action::PushTerminal(t)));
                    actions.push(Action::ExitMacroDef)
                }
            }
            actions
        }
    }

    #[derive(Clone)]
    struct TokenSeq(Vec<usize>);

    fn token_seq(ids: impl Borrow<[usize]>) -> TokenSeq {
        TokenSeq(ids.borrow().iter().cloned().collect())
    }

    impl TokenSeq {
        fn into_actions(self) -> Vec<Action> {
            self.0
                .into_iter()
                .map(|id| Action::PushTerminal(id))
                .collect()
        }
    }

    #[test]
    fn parse_nullary_instruction_followed_by_eol() {
        assert_eq_actions(
            input_tokens![daa @ Command(()), Eol],
            file([unlabeled(command("daa", [])), unlabeled(empty())]),
        )
    }

    #[test]
    fn parse_unary_instruction() {
        assert_eq_actions(
            input_tokens![db @ Command(()), my_ptr @ Ident(())],
            file([unlabeled(command("db", [ident("my_ptr")]))]),
        )
    }

    fn ident(id: impl Into<TokenRef>) -> SymExpr {
        SymExpr::Ident(id.into())
    }

    fn literal(id: impl Into<TokenRef>) -> SymExpr {
        SymExpr::Literal(id.into())
    }

    fn parentheses(
        open_id: impl Into<TokenRef>,
        expr: SymExpr,
        close_id: impl Into<TokenRef>,
    ) -> SymExpr {
        SymExpr::Parentheses(open_id.into(), Box::new(expr), close_id.into())
    }

    #[derive(Clone)]
    enum SymExpr {
        Ident(TokenRef),
        Literal(TokenRef),
        Parentheses(TokenRef, Box<SymExpr>, TokenRef),
    }

    impl SymExpr {
        fn into_actions(self, input: &InputTokens) -> Vec<Action> {
            match self {
                SymExpr::Ident(ident) => vec![Action::PushExprAtom(ExprAtom::Ident(SymIdent(
                    ident.resolve(input),
                )))],
                SymExpr::Literal(literal) => vec![Action::PushExprAtom(ExprAtom::Literal(
                    SymLiteral(literal.resolve(input)),
                ))],
                SymExpr::Parentheses(_, expr, _) => {
                    let mut actions = expr.into_actions(input);
                    actions.push(Action::ApplyExprOperator(ExprOperator::Parentheses));
                    actions
                }
            }
        }
    }

    #[test]
    fn parse_binary_instruction() {
        assert_eq_actions(
            input_tokens![Command(()), Ident(()), Comma, Literal(())],
            file([unlabeled(command(0, [ident(1), literal(3)]))]),
        );
    }

    #[test]
    fn parse_two_instructions() {
        let tokens = input_tokens![
            Command(()),
            Ident(()),
            Comma,
            Literal(()),
            Eol,
            ld @ Command(()),
            a @ Literal(()),
            Comma,
            some_const @ Ident(()),
        ];
        let expected = file([
            unlabeled(command(0, [ident(1), literal(3)])),
            unlabeled(command("ld", [literal("a"), ident("some_const")])),
        ]);
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_two_instructions_separated_by_blank_line() {
        let tokens = input_tokens![
            Command(()),
            Literal(()),
            Comma,
            Ident(()),
            Eol,
            Eol,
            Command(()),
            Ident(()),
            Comma,
            Literal(()),
        ];
        let expected = file([
            unlabeled(command(0, [literal(1), ident(3)])),
            unlabeled(empty()),
            unlabeled(command(6, [ident(7), literal(9)])),
        ]);
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_empty_macro_definition() {
        let tokens = input_tokens![Ident(()), Colon, Macro, Eol, Endm];
        let expected_actions = file([labeled(0, macro_def([], []))]);
        assert_eq_actions(tokens, expected_actions);
    }

    #[test]
    fn parse_macro_definition_with_instruction() {
        let tokens = input_tokens![Ident(()), Colon, Macro, Eol, Command(()), Eol, Endm];
        let expected_actions = file([labeled(0, macro_def([], [4, 5]))]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_nonempty_macro_def_with_two_params() {
        let tokens = input_tokens![
            Ident(()),
            Colon,
            Macro,
            Ident(()),
            Comma,
            Ident(()),
            Eol,
            Command(()),
            Eol,
            Endm,
        ];
        let expected = file([labeled(0, macro_def([3, 5], [7, 8]))]);
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_label() {
        let tokens = input_tokens![Ident(()), Colon, Eol];
        let expected_actions = file([labeled(0, empty()), unlabeled(empty())]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_labeled_instruction() {
        let tokens = input_tokens![Ident(()), Colon, Command(()), Eol];
        let expected = file([labeled(0, command(2, [])), unlabeled(empty())]);
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_deref_operand() {
        let tokens = input_tokens![
            jp @ Command(()),
            open @ OpeningParenthesis,
            hl @ Literal(()),
            close @ ClosingParenthesis,
        ];
        let expected = file([unlabeled(command(
            "jp",
            [parentheses("open", literal("hl"), "close")],
        ))]);
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_nullary_macro_invocation() {
        let tokens = input_tokens![Ident(())];
        let expected_actions = file([unlabeled(invoke(0, []))]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_invocation() {
        let tokens = input_tokens![Ident(()), Literal(())];
        let expected_actions = file([unlabeled(invoke(0, [token_seq([1])]))]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_invocation_with_multiple_terminals() {
        let tokens = input_tokens![Ident(()), Literal(()), Literal(()), Literal(())];
        let expected_actions = file([unlabeled(invoke(0, [token_seq([1, 2, 3])]))]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_binary_macro_invocation_with_multiple_terminals() {
        let tokens = input_tokens![
            Ident(()),
            Literal(()),
            Literal(()),
            Comma,
            Literal(()),
            Literal(()),
            Literal(()),
        ];
        let expected_actions = file([unlabeled(invoke(
            0,
            [token_seq([1, 2]), token_seq([4, 5, 6])],
        ))]);
        assert_eq_actions(tokens, expected_actions)
    }

    fn line_error(
        message_ctor: MessageCtor,
        ranges: impl Borrow<[&'static str]>,
        highlight: impl Into<TokenRef>,
    ) -> Option<LineBody> {
        Some(
            SymDiagnostic {
                message_ctor,
                ranges: ranges
                    .borrow()
                    .iter()
                    .map(|s| TokenRef::from(*s).into())
                    .collect(),
                highlight: highlight.into().into(),
            }.into(),
        )
    }

    fn arg_error(
        message_ctor: MessageCtor,
        ranges: impl Borrow<[&'static str]>,
        highlight: impl Into<TokenRef>,
    ) -> SymDiagnostic {
        SymDiagnostic {
            message_ctor,
            ranges: ranges
                .borrow()
                .iter()
                .map(|s| TokenRef::from(*s).into())
                .collect(),
            highlight: highlight.into().into(),
        }
    }

    #[test]
    fn diagnose_stmt_starting_with_literal() {
        assert_eq_actions(
            input_tokens![a @ Literal(())],
            file([unlabeled(line_error(unexpected_token, ["a"], "a"))]),
        )
    }

    #[test]
    fn diagnose_missing_comma_in_arg_list() {
        assert_eq_actions(
            input_tokens![Command(()), Literal(()), unexpected @ Literal(())],
            file([unlabeled(malformed_command(
                0,
                [literal(1)],
                arg_error(unexpected_token, ["unexpected"], "unexpected"),
            ))]),
        )
    }

    fn unexpected_token(ranges: Vec<SymRange<usize>>) -> Message<SymRange<usize>> {
        Message::UnexpectedToken {
            token: ranges.into_iter().next().unwrap(),
        }
    }
}
