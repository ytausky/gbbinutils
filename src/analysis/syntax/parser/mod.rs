use super::SimpleToken::*;
use super::*;
use crate::diag::span::StripSpan;
use crate::diag::{EmitDiag, Message};

macro_rules! bump {
    ($parser:expr) => {
        $parser.state.token = $parser.state.remaining.next().unwrap()
    };
}

#[macro_use]
#[cfg(test)]
mod mock;
mod expr;

type TokenKind = Token<(), ()>;

impl Copy for TokenKind {}

impl<I, L> Token<I, L> {
    fn kind(&self) -> TokenKind {
        use self::Token::*;
        match *self {
            Ident(_) => Ident(()),
            Label(_) => Label(()),
            Literal(_) => Literal(()),
            Simple(simple) => Simple(simple),
        }
    }
}

const LINE_FOLLOW_SET: &[TokenKind] = &[Token::Simple(Eol), Token::Simple(Eof)];

pub(in crate::analysis) fn parse_src<Id, L, E, I, F, S>(mut tokens: I, context: F) -> F
where
    I: Iterator<Item = (Result<Token<Id, L>, E>, S)>,
    F: FileContext<Id, L, S>,
    S: Clone,
{
    let Parser {
        state: ParserState { token, .. },
        context,
        ..
    } = Parser::new(&mut tokens, context).parse_file();
    assert_eq!(
        token.0.ok().as_ref().map(Token::kind),
        Some(Token::Simple(SimpleToken::Eof))
    );
    context
}

struct Parser<'a, T, I: 'a, C> {
    state: ParserState<'a, T, I>,
    context: C,
}

struct ParserState<'a, T, I> {
    token: T,
    remaining: &'a mut I,
    recovery: Option<RecoveryState>,
}

enum RecoveryState {
    DiagnosedEof,
}

impl<'a, T, I: Iterator<Item = T>, C> Parser<'a, T, I, C> {
    fn new(tokens: &'a mut I, context: C) -> Self {
        Self::from_state(
            ParserState {
                token: tokens.next().unwrap(),
                remaining: tokens,
                recovery: None,
            },
            context,
        )
    }
}

impl<'a, T, I, C> Parser<'a, T, I, C> {
    fn from_state(state: ParserState<'a, T, I>, context: C) -> Self {
        Parser { state, context }
    }

    fn change_context<D, F: FnOnce(C) -> D>(self, f: F) -> Parser<'a, T, I, D> {
        Parser::from_state(self.state, f(self.context))
    }
}

impl<'a, Id, L, E, S, I, A> Parser<'a, (Result<Token<Id, L>, E>, S), I, A> {
    fn token_kind(&self) -> Option<TokenKind> {
        self.state.token.0.as_ref().ok().map(Token::kind)
    }

    fn token_is_in(&self, kinds: &[TokenKind]) -> bool {
        match &self.state.token.0 {
            Ok(token) => kinds.iter().any(|x| *x == token.kind()),
            Err(_) => false,
        }
    }
}

impl<'a, Id, L, E, I, C, S> Parser<'a, (Result<Token<Id, L>, E>, S), I, C>
where
    I: Iterator<Item = (Result<Token<Id, L>, E>, S)>,
    C: FileContext<Id, L, S>,
    S: Clone,
{
    fn parse_file(mut self) -> Self {
        while self.token_kind() != Some(Token::Simple(SimpleToken::Eof)) {
            self = self.parse_stmt()
        }
        self
    }

    fn parse_stmt(mut self) -> Self {
        if let (Ok(Token::Label(label)), span) = self.state.token {
            bump!(self);
            let mut parser = self.change_context(|c| c.enter_labeled_stmt((label, span)));
            if parser.token_kind() == Some(LParen.into()) {
                bump!(parser);
                parser = parser.parse_terminated_list(
                    Comma.into(),
                    &[RParen.into()],
                    Parser::parse_param,
                );
                if parser.token_kind() == Some(RParen.into()) {
                    bump!(parser)
                } else {
                    parser = parser.diagnose_unexpected_token();
                    return parser
                        .change_context(ParamsContext::next)
                        .change_context(StmtContext::exit);
                }
            }
            parser.change_context(ParamsContext::next)
        } else {
            self.change_context(FileContext::enter_unlabeled_stmt)
        }
        .parse_unlabeled_stmt()
        .change_context(StmtContext::exit)
    }
}

impl<'a, Id, L, E, I, C, S> Parser<'a, (Result<Token<Id, L>, E>, S), I, C>
where
    I: Iterator<Item = (Result<Token<Id, L>, E>, S)>,
    C: StmtContext<Id, L, S>,
    S: Clone,
{
    fn parse_unlabeled_stmt(mut self) -> Self {
        loop {
            return match self.state.token {
                (Ok(Token::Simple(SimpleToken::Eol)), _) => {
                    bump!(self);
                    continue;
                }
                (Ok(Token::Label(_)), _) | (Ok(Token::Simple(SimpleToken::Eof)), _) => self,
                (Ok(Token::Ident(ident)), span) => {
                    bump!(self);
                    self.parse_key(ident, span)
                }
                (_, span) => {
                    bump!(self);
                    let stripped = self.strip_span(&span);
                    self.emit_diag(Message::UnexpectedToken { token: stripped }.at(span));
                    self
                }
            };
        }
    }

    fn parse_key(mut self, key: Id, span: S) -> Self {
        match self.context.key_lookup(key, span) {
            Production::BuiltinInstr(context) => Parser::from_state(self.state, context)
                .parse_argument_list()
                .change_context(BuiltinInstrContext::exit),
            Production::MacroCall(context) => Parser::from_state(self.state, context)
                .parse_macro_call()
                .change_context(MacroCallContext::exit),
            Production::MacroDef(context) => Parser::from_state(self.state, context)
                .parse_macro_def()
                .change_context(TokenSeqContext::exit),
            Production::Error(context) => {
                self.context = context;
                while !self.token_is_in(LINE_FOLLOW_SET) {
                    bump!(self)
                }
                self
            }
        }
    }
}

impl<'a, Id, L, E, I, C, S> Parser<'a, (Result<Token<Id, L>, E>, S), I, C>
where
    I: Iterator<Item = (Result<Token<Id, L>, E>, S)>,
    C: BuiltinInstrContext<S, Ident = Id, Literal = L>,
    S: Clone,
{
    fn parse_argument_list(self) -> Self {
        self.parse_terminated_list(Comma.into(), LINE_FOLLOW_SET, Parser::parse_argument)
    }

    fn parse_argument(self) -> Self {
        self.change_context(BuiltinInstrContext::add_argument)
            .parse()
            .change_context(FinalContext::exit)
    }
}

impl<'a, Id, L, E, I, C, S> Parser<'a, (Result<Token<Id, L>, E>, S), I, C>
where
    I: Iterator<Item = (Result<Token<Id, L>, E>, S)>,
    C: ParamsContext<Id, S>,
    S: Clone,
{
    fn parse_param(mut self) -> Self {
        match self.state.token.0 {
            Ok(Token::Ident(ident)) => {
                self.context.add_parameter((ident, self.state.token.1));
                bump!(self)
            }
            _ => self = self.diagnose_unexpected_token(),
        };
        self
    }
}

impl<'a, Id, L, E, I, C, S> Parser<'a, (Result<Token<Id, L>, E>, S), I, C>
where
    I: Iterator<Item = (Result<Token<Id, L>, E>, S)>,
    C: MacroCallContext<S, Token = Token<Id, L>>,
    S: Clone,
{
    fn parse_macro_call(self) -> Self {
        self.parse_macro_arg_list()
    }

    fn parse_macro_arg_list(self) -> Self {
        self.parse_terminated_list(Comma.into(), LINE_FOLLOW_SET, |p| {
            let mut parser = p.change_context(MacroCallContext::enter_macro_arg);
            loop {
                match parser.state.token {
                    (Ok(Token::Simple(Comma)), _)
                    | (Ok(Token::Simple(Eol)), _)
                    | (Ok(Token::Simple(Eof)), _) => break,
                    (Ok(other), span) => {
                        bump!(parser);
                        parser.context.push_token((other, span))
                    }
                    (Err(_), _) => unimplemented!(),
                }
            }
            parser.change_context(TokenSeqContext::exit)
        })
    }
}

impl<'a, Id, L, E, I, C, S> Parser<'a, (Result<Token<Id, L>, E>, S), I, C>
where
    I: Iterator<Item = (Result<Token<Id, L>, E>, S)>,
    C: TokenSeqContext<S, Token = Token<Id, L>>,
    S: Clone,
{
    fn parse_macro_def(mut self) -> Self {
        if !self.token_is_in(LINE_FOLLOW_SET) {
            self = self.diagnose_unexpected_token();
            while !self.token_is_in(LINE_FOLLOW_SET) {
                bump!(self)
            }
        }
        if self.token_kind() == Some(Eol.into()) {
            bump!(self);
            loop {
                match self.state.token {
                    (Ok(Token::Simple(Endm)), _) => {
                        self.context
                            .push_token((Eof.into(), self.state.token.1.clone()));
                        bump!(self);
                        break;
                    }
                    (Ok(Token::Simple(Eof)), _) => {
                        self = self.diagnose_unexpected_token();
                        break;
                    }
                    (Ok(other), span) => {
                        self.context.push_token((other, span));
                        bump!(self);
                    }
                    (Err(_), _) => unimplemented!(),
                }
            }
        } else {
            assert_eq!(self.token_kind(), Some(Eof.into()));
            self = self.diagnose_unexpected_token();
        }
        self
    }
}

impl<'a, Id, L, E, I, C, S> Parser<'a, (Result<Token<Id, L>, E>, S), I, C>
where
    I: Iterator<Item = (Result<Token<Id, L>, E>, S)>,
    C: Diagnostics<S>,
    S: Clone,
{
    fn parse_terminated_list<P>(
        mut self,
        delimiter: TokenKind,
        terminators: &[TokenKind],
        parser: P,
    ) -> Self
    where
        P: FnMut(Self) -> Self,
    {
        self = self.parse_list(delimiter, terminators, parser);
        if !self.token_is_in(terminators) {
            self = self.diagnose_unexpected_token();
            while !self.token_is_in(terminators) && self.token_kind() != Some(Token::Simple(Eof)) {
                bump!(self);
            }
        }
        self
    }

    fn parse_list<P>(self, delimiter: TokenKind, terminators: &[TokenKind], mut parser: P) -> Self
    where
        P: FnMut(Self) -> Self,
    {
        if self.token_is_in(terminators) {
            self
        } else {
            self.parse_nonempty_list(delimiter, &mut parser)
        }
    }

    fn parse_nonempty_list<P>(mut self, delimiter: TokenKind, parser: &mut P) -> Self
    where
        P: FnMut(Self) -> Self,
    {
        self = parser(self);
        while self.token_kind() == Some(delimiter) {
            bump!(self);
            self = parser(self);
        }
        self
    }

    fn diagnose_unexpected_token(mut self) -> Self {
        if self.token_kind() == Some(Token::Simple(Eof)) {
            if self.state.recovery.is_none() {
                self.emit_diag(Message::UnexpectedEof.at(self.state.token.1.clone()));
                self.state.recovery = Some(RecoveryState::DiagnosedEof)
            }
        } else {
            let token = self.state.token.1;
            bump!(self);
            let stripped = self.strip_span(&token);
            self.emit_diag(Message::UnexpectedToken { token: stripped }.at(token))
        }
        self
    }
}

delegate_diagnostics! {
    {'a, T, I, C: Diagnostics<S>, S}, Parser<'a, T, I, C>, {context}, C, S
}

#[cfg(test)]
mod tests {
    use super::mock::*;
    use super::Token::*;
    use super::*;

    use crate::diag::Message;

    use std::borrow::Borrow;

    #[test]
    fn parse_empty_src() {
        assert_eq_actions(input_tokens![], [])
    }

    #[test]
    fn parse_empty_stmt() {
        assert_eq_actions(input_tokens![Eol], [unlabeled(empty())])
    }

    fn assert_eq_actions(input: InputTokens, expected: impl Borrow<[FileAction<MockSpan>]>) {
        let mut parsing_context = FileActionCollector::new();
        parsing_context = parse_src(with_spans(&input.tokens), parsing_context);
        assert_eq!(parsing_context.actions, expected.borrow())
    }

    #[test]
    fn parse_nullary_instruction() {
        assert_eq_actions(
            input_tokens![nop @ Ident(IdentKind::BuiltinInstr)],
            [unlabeled(builtin_instr("nop", []))],
        )
    }

    #[test]
    fn parse_nullary_instruction_after_eol() {
        assert_eq_actions(
            input_tokens![Eol, nop @ Ident(IdentKind::BuiltinInstr)],
            [unlabeled(builtin_instr("nop", []))],
        )
    }

    #[test]
    fn parse_nullary_instruction_followed_by_eol() {
        assert_eq_actions(
            input_tokens![daa @ Ident(IdentKind::BuiltinInstr), Eol],
            [unlabeled(builtin_instr("daa", [])), unlabeled(empty())],
        )
    }

    #[test]
    fn parse_unary_instruction() {
        assert_eq_actions(
            input_tokens![db @ Ident(IdentKind::BuiltinInstr), my_ptr @ Ident(IdentKind::Other)],
            [unlabeled(builtin_instr("db", [expr().ident("my_ptr")]))],
        )
    }

    #[test]
    fn parse_binary_instruction() {
        assert_eq_actions(
            input_tokens![
                Ident(IdentKind::BuiltinInstr),
                Ident(IdentKind::Other),
                Comma,
                Literal(())
            ],
            [unlabeled(builtin_instr(
                0,
                [expr().ident(1), expr().literal(3)],
            ))],
        );
    }

    #[test]
    fn parse_two_instructions() {
        let tokens = input_tokens![
            Ident(IdentKind::BuiltinInstr),
            Ident(IdentKind::Other),
            Comma,
            Literal(()),
            Eol,
            ld @ Ident(IdentKind::BuiltinInstr),
            a @ Literal(()),
            Comma,
            some_const @ Ident(IdentKind::Other),
        ];
        let expected = [
            unlabeled(builtin_instr(0, [expr().ident(1), expr().literal(3)])),
            unlabeled(builtin_instr(
                "ld",
                [expr().literal("a"), expr().ident("some_const")],
            )),
        ];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_two_instructions_separated_by_blank_line() {
        let tokens = input_tokens![
            Ident(IdentKind::BuiltinInstr),
            Literal(()),
            Comma,
            Ident(IdentKind::Other),
            Eol,
            Eol,
            Ident(IdentKind::BuiltinInstr),
            Ident(IdentKind::Other),
            Comma,
            Literal(()),
        ];
        let expected = [
            unlabeled(builtin_instr(0, [expr().literal(1), expr().ident(3)])),
            unlabeled(builtin_instr(6, [expr().ident(7), expr().literal(9)])),
        ];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_empty_macro_definition() {
        let tokens = input_tokens![
            Label(IdentKind::Other),
            Ident(IdentKind::MacroKeyword),
            Eol,
            Endm,
        ];
        let expected_actions = [labeled(0, vec![], macro_def(1, Vec::new(), 3))];
        assert_eq_actions(tokens, expected_actions);
    }

    #[test]
    fn parse_macro_definition_with_instruction() {
        let tokens = input_tokens![
            Label(IdentKind::Other),
            Ident(IdentKind::MacroKeyword),
            Eol,
            Ident(IdentKind::BuiltinInstr),
            Eol,
            Endm,
        ];
        let expected_actions = [labeled(
            0,
            vec![],
            macro_def(1, tokens.token_seq([3, 4]), 5),
        )];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_nonempty_macro_def_with_two_params() {
        let tokens = input_tokens![
            l @ Label(IdentKind::Other),
            LParen,
            p1 @ Ident(IdentKind::Other),
            Comma,
            p2 @ Ident(IdentKind::Other),
            RParen,
            key @ Ident(IdentKind::MacroKeyword),
            Eol,
            t1 @ Ident(IdentKind::BuiltinInstr),
            t2 @ Eol,
            endm @ Endm,
        ];
        let expected = [labeled(
            "l",
            ["p1".into(), "p2".into()],
            macro_def("key", tokens.token_seq(["t1", "t2"]), "endm"),
        )];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_label() {
        let tokens = input_tokens![Label(IdentKind::Other), Eol];
        let expected_actions = [labeled(0, vec![], empty())];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_two_consecutive_labels() {
        let tokens = input_tokens![Label(IdentKind::Other), Eol, Label(IdentKind::Other)];
        let expected = [labeled(0, vec![], empty()), labeled(2, vec![], empty())];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_labeled_instruction() {
        let tokens = input_tokens![Label(IdentKind::Other), Ident(IdentKind::BuiltinInstr), Eol];
        let expected = [labeled(0, vec![], builtin_instr(1, [])), unlabeled(empty())];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_labeled_command_with_eol_separators() {
        let tokens = input_tokens![
            Label(IdentKind::Other),
            Eol,
            Eol,
            Ident(IdentKind::BuiltinInstr)
        ];
        let expected = [labeled(0, vec![], builtin_instr(3, []))];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_deref_operand() {
        let tokens = input_tokens![
            jp @ Ident(IdentKind::BuiltinInstr),
            open @ LParen,
            hl @ Literal(()),
            close @ RParen,
        ];
        let expected = [unlabeled(builtin_instr(
            "jp",
            [expr().literal("hl").parentheses("open", "close")],
        ))];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_nullary_macro_call() {
        let tokens = input_tokens![Ident(IdentKind::MacroName)];
        let expected_actions = [unlabeled(call_macro(0, []))];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_call() {
        let tokens = input_tokens![Ident(IdentKind::MacroName), Literal(())];
        let expected_actions = [unlabeled(call_macro(0, [tokens.token_seq([1])]))];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_call_with_multiple_terminals() {
        let tokens = input_tokens![
            Ident(IdentKind::MacroName),
            Literal(()),
            Literal(()),
            Literal(())
        ];
        let expected_actions = [unlabeled(call_macro(0, [tokens.token_seq([1, 2, 3])]))];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_binary_macro_call_with_multiple_terminals() {
        let tokens = input_tokens![
            Ident(IdentKind::MacroName),
            Literal(()),
            Literal(()),
            Comma,
            Literal(()),
            Literal(()),
            Literal(()),
        ];
        let expected_actions = [unlabeled(call_macro(
            0,
            [tokens.token_seq([1, 2]), tokens.token_seq([4, 5, 6])],
        ))];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_sum_arg() {
        let tokens = input_tokens![
            Ident(IdentKind::BuiltinInstr),
            x @ Ident(IdentKind::Other),
            plus @ Plus,
            y @ Literal(()),
        ];
        let expected_actions = [unlabeled(builtin_instr(
            0,
            [expr().ident("x").literal("y").plus("plus")],
        ))];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn diagnose_stmt_starting_with_literal() {
        let token: MockSpan = TokenRef::from("a").into();
        assert_eq_actions(
            input_tokens![a @ Literal(())],
            [unlabeled(stmt_error(
                Message::UnexpectedToken { token },
                "a",
            ))],
        )
    }

    #[test]
    fn diagnose_missing_comma_in_arg_list() {
        let span: MockSpan = TokenRef::from("unexpected").into();
        assert_eq_actions(
            input_tokens![Ident(IdentKind::BuiltinInstr), Literal(()), unexpected @ Literal(())],
            [unlabeled(malformed_builtin_instr(
                0,
                [expr().literal(1)],
                Message::UnexpectedToken {
                    token: span.clone(),
                }
                .at(span)
                .into(),
            ))],
        )
    }

    #[test]
    fn diagnose_eof_in_param_list() {
        assert_eq_actions(
            input_tokens![label @ Label(IdentKind::Other), LParen, eof @ Eof],
            [FileAction::Stmt {
                label: Some((
                    (
                        SymIdent(IdentKind::Other, "label".into()),
                        TokenRef::from("label").into(),
                    ),
                    vec![ParamsAction::EmitDiag(arg_error(
                        Message::UnexpectedEof,
                        "eof",
                    ))],
                )),
                actions: vec![],
            }],
        )
    }

    #[test]
    fn diagnose_eof_in_macro_body() {
        assert_eq_actions(
            input_tokens![Ident(IdentKind::MacroKeyword), Eol, eof @ Eof],
            [unlabeled(malformed_macro_def(
                0,
                Vec::new(),
                arg_error(Message::UnexpectedEof, "eof"),
            ))],
        )
    }

    #[test]
    fn diagnose_unmatched_parentheses() {
        assert_eq_actions(
            input_tokens![Ident(IdentKind::BuiltinInstr), paren @ LParen, Literal(())],
            [unlabeled(builtin_instr(
                0,
                [expr()
                    .literal(2)
                    .error(Message::UnmatchedParenthesis, TokenRef::from("paren"))],
            ))],
        )
    }

    #[test]
    fn recover_from_unexpected_token_in_expr() {
        let paren_span: MockSpan = TokenRef::from("paren").into();
        assert_eq_actions(
            input_tokens![
                Ident(IdentKind::BuiltinInstr),
                paren @ RParen,
                Plus,
                Ident(IdentKind::Other),
                Eol,
                nop @ Ident(IdentKind::BuiltinInstr)
            ],
            [
                unlabeled(builtin_instr(
                    0,
                    [expr().error(
                        Message::UnexpectedToken {
                            token: paren_span.clone(),
                        },
                        paren_span,
                    )],
                )),
                unlabeled(builtin_instr("nop", [])),
            ],
        )
    }

    #[test]
    fn diagnose_unmatched_parenthesis_at_eol() {
        assert_eq_actions(
            input_tokens![Ident(IdentKind::BuiltinInstr), LParen, Eol],
            [
                unlabeled(builtin_instr(
                    0,
                    [expr().error(Message::UnmatchedParenthesis, TokenRef::from(1))],
                )),
                unlabeled(empty()),
            ],
        )
    }

    #[test]
    fn diagnose_unexpected_token_param_list() {
        let span: MockSpan = TokenRef::from("lit").into();
        assert_eq_actions(
            input_tokens![
                label @ Label(IdentKind::Other),
                LParen,
                lit @ Literal(()),
                RParen,
                key @ Ident(IdentKind::MacroKeyword),
                Eol,
                endm @ Endm
            ],
            [FileAction::Stmt {
                label: Some((
                    (
                        SymIdent(IdentKind::Other, "label".into()),
                        TokenRef::from("label").into(),
                    ),
                    vec![ParamsAction::EmitDiag(
                        Message::UnexpectedToken {
                            token: span.clone(),
                        }
                        .at(span)
                        .into(),
                    )],
                )),
                actions: vec![StmtAction::MacroDef {
                    keyword: (
                        SymIdent(IdentKind::MacroKeyword, "key".into()),
                        TokenRef::from("key").into(),
                    ),
                    body: vec![TokenSeqAction::PushToken((
                        Eof.into(),
                        TokenRef::from("endm").into(),
                    ))],
                }],
            }],
        )
    }

    #[test]
    fn diagnose_unexpected_token_after_macro_keyword() {
        let tokens = input_tokens![
            label @ Label(IdentKind::Other),
            key @ Ident(IdentKind::MacroKeyword),
            unexpected @ Ident(IdentKind::Other),
            Eol,
            t1 @ Ident(IdentKind::BuiltinInstr),
            t2 @ Eol,
            t3 @ Endm,
        ];
        let unexpected = TokenRef::from("unexpected");
        let mut body = vec![TokenSeqAction::EmitDiag(arg_error(
            Message::UnexpectedToken {
                token: unexpected.clone().into(),
            },
            unexpected,
        ))];
        body.extend(tokens.token_seq(["t1", "t2"]));
        body.push(push_token(Eof, "t3"));
        let expected = [labeled(
            "label",
            vec![],
            vec![StmtAction::MacroDef {
                keyword: (
                    SymIdent(IdentKind::MacroKeyword, "key".into()),
                    TokenRef::from("key").into(),
                ),
                body,
            }],
        )];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn recover_from_other_ident_as_key() {
        let tokens = input_tokens![
            Ident(IdentKind::Other),
            Ident(IdentKind::Other),
            Eol,
            nop @ Ident(IdentKind::BuiltinInstr),
        ];
        let expected = [unlabeled(vec![]), unlabeled(builtin_instr("nop", []))];
        assert_eq_actions(tokens, expected)
    }
}
