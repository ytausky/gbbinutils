use super::Sigil::*;
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
            Sigil(sigil) => Sigil(sigil),
        }
    }
}

const LINE_FOLLOW_SET: &[TokenKind] = &[Token::Sigil(Eol), Token::Sigil(Eos)];

pub(in crate::analysis) fn parse_src<I, L, E, R, A, S>(mut tokens: R, actions: A) -> A::Next
where
    R: Iterator<Item = (Result<Token<I, L>, E>, S)>,
    A: TokenStreamActions<I, L, S>,
    S: Clone,
{
    let Parser {
        state: ParserState { token, .. },
        actions,
        ..
    } = Parser::new(&mut tokens, actions).parse_token_stream();
    assert_eq!(
        token.0.ok().as_ref().map(Token::kind),
        Some(Token::Sigil(Sigil::Eos))
    );
    actions.act_on_eos(token.1)
}

struct Parser<'a, T, R: 'a, A> {
    state: ParserState<'a, T, R>,
    actions: A,
}

struct ParserState<'a, T, R> {
    token: T,
    remaining: &'a mut R,
    parsed_eos: bool,
    recovery: Option<RecoveryState>,
}

enum RecoveryState {
    DiagnosedEof,
}

impl<'a, T, R: Iterator<Item = T>, A> Parser<'a, T, R, A> {
    fn new(tokens: &'a mut R, actions: A) -> Self {
        Self::from_state(
            ParserState {
                token: tokens.next().unwrap(),
                remaining: tokens,
                parsed_eos: false,
                recovery: None,
            },
            actions,
        )
    }
}

impl<'a, T, R, A> Parser<'a, T, R, A> {
    fn from_state(state: ParserState<'a, T, R>, actions: A) -> Self {
        Parser { state, actions }
    }

    fn change_context<D, F: FnOnce(A) -> D>(self, f: F) -> Parser<'a, T, R, D> {
        Parser::from_state(self.state, f(self.actions))
    }
}

impl<'a, I, L, E, S, R, A> Parser<'a, (Result<Token<I, L>, E>, S), R, A> {
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

impl<'a, I, L, E, T, C, S> Parser<'a, (Result<Token<I, L>, E>, S), T, C>
where
    T: Iterator<Item = (Result<Token<I, L>, E>, S)>,
    C: TokenStreamActions<I, L, S>,
    S: Clone,
{
    fn parse_token_stream(mut self) -> Self {
        while !self.state.parsed_eos {
            self = match self.actions.will_parse_line() {
                LineRule::InstrLine(actions) => {
                    Parser::from_state(self.state, actions).parse_instr_line()
                }
                LineRule::TokenLine(actions) => {
                    Parser::from_state(self.state, actions).parse_token_line()
                }
            }
        }
        self
    }
}

type NextParser<'a, I, L, E, R, A, S> =
    Parser<'a, (Result<Token<I, L>, E>, S), R, <A as LineFinalizer<S>>::Next>;

impl<'a, I, L, E, R, A, S> Parser<'a, (Result<Token<I, L>, E>, S), R, A>
where
    R: Iterator<Item = (Result<Token<I, L>, E>, S)>,
    A: InstrLineActions<I, L, S>,
    S: Clone,
{
    fn parse_instr_line(mut self) -> NextParser<'a, I, L, E, R, A, S> {
        if let (Ok(Token::Label(label)), span) = self.state.token {
            bump!(self);
            let mut parser = self.change_context(|c| c.will_parse_label((label, span)));
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
                        .change_context(LabelActions::did_parse_label)
                        .parse_line_terminator();
                }
            }
            parser
                .change_context(LabelActions::did_parse_label)
                .parse_unlabeled_stmt()
        } else {
            self.parse_unlabeled_stmt()
        }
    }
}

impl<'a, I, L, E, R, A, S> Parser<'a, (Result<Token<I, L>, E>, S), R, A>
where
    R: Iterator<Item = (Result<Token<I, L>, E>, S)>,
    A: InstrActions<I, L, S>,
    S: Clone,
{
    fn parse_unlabeled_stmt(mut self) -> NextParser<'a, I, L, E, R, A, S> {
        match self.state.token {
            (Ok(Token::Sigil(Sigil::Eol)), _) | (Ok(Token::Sigil(Sigil::Eos)), _) => {
                self.parse_line_terminator()
            }
            (Ok(Token::Ident(ident)), span) => {
                bump!(self);
                self.parse_key(ident, span)
            }
            (_, span) => {
                bump!(self);
                let stripped = self.strip_span(&span);
                self.emit_diag(Message::UnexpectedToken { token: stripped }.at(span));
                self.parse_line_terminator()
            }
        }
    }

    fn parse_key(self, key: I, span: S) -> NextParser<'a, I, L, E, R, A, S> {
        match self.actions.will_parse_instr(key, span) {
            InstrRule::BuiltinInstr(actions) => Parser::from_state(self.state, actions)
                .parse_argument_list()
                .change_context(InstrFinalizer::did_parse_instr)
                .parse_line_terminator(),
            InstrRule::MacroInstr(actions) => Parser::from_state(self.state, actions)
                .parse_macro_call()
                .change_context(InstrFinalizer::did_parse_instr)
                .parse_line_terminator(),
            InstrRule::Error(actions) => {
                let mut parser = Parser::from_state(self.state, actions.did_parse_instr());
                while !parser.token_is_in(LINE_FOLLOW_SET) {
                    bump!(parser)
                }
                parser.parse_line_terminator()
            }
        }
    }
}

impl<'a, I, L, E, R, A, S> Parser<'a, (Result<Token<I, L>, E>, S), R, A>
where
    R: Iterator<Item = (Result<Token<I, L>, E>, S)>,
    A: LineFinalizer<S>,
    S: Clone,
{
    fn parse_line_terminator(mut self) -> NextParser<'a, I, L, E, R, A, S> {
        let span = match &self.state.token {
            (Ok(Token::Sigil(Sigil::Eol)), _) => {
                let span = self.state.token.1;
                bump!(self);
                span
            }
            (Ok(Token::Sigil(Sigil::Eos)), span) => {
                self.state.parsed_eos = true;
                span.clone()
            }
            (Ok(_), _) => panic!("expected line terminator"),
            (Err(_), _) => unimplemented!(),
        };
        self.change_context(|actions| actions.did_parse_line(span))
    }
}

impl<'a, I, L, E, R, A, S> Parser<'a, (Result<Token<I, L>, E>, S), R, A>
where
    R: Iterator<Item = (Result<Token<I, L>, E>, S)>,
    A: BuiltinInstrActions<S, Ident = I, Literal = L>,
    S: Clone,
{
    fn parse_argument_list(self) -> Self {
        self.parse_terminated_list(Comma.into(), LINE_FOLLOW_SET, Parser::parse_argument)
    }

    fn parse_argument(self) -> Self {
        self.change_context(BuiltinInstrActions::will_parse_arg)
            .parse()
            .change_context(ArgFinalizer::did_parse_arg)
    }
}

impl<'a, I, L, E, R, A, S> Parser<'a, (Result<Token<I, L>, E>, S), R, A>
where
    R: Iterator<Item = (Result<Token<I, L>, E>, S)>,
    A: LabelActions<I, S>,
    S: Clone,
{
    fn parse_param(mut self) -> Self {
        match self.state.token.0 {
            Ok(Token::Ident(ident)) => {
                self.actions.act_on_param((ident, self.state.token.1));
                bump!(self)
            }
            _ => self = self.diagnose_unexpected_token(),
        };
        self
    }
}

impl<'a, I, L, E, R, A, S> Parser<'a, (Result<Token<I, L>, E>, S), R, A>
where
    R: Iterator<Item = (Result<Token<I, L>, E>, S)>,
    A: MacroInstrActions<S, Token = Token<I, L>>,
    S: Clone,
{
    fn parse_macro_call(self) -> Self {
        self.parse_macro_arg_list()
    }

    fn parse_macro_arg_list(self) -> Self {
        self.parse_terminated_list(Comma.into(), LINE_FOLLOW_SET, |p| {
            let mut parser = p.change_context(MacroInstrActions::will_parse_macro_arg);
            loop {
                match parser.state.token {
                    (Ok(Token::Sigil(Comma)), _)
                    | (Ok(Token::Sigil(Eol)), _)
                    | (Ok(Token::Sigil(Eos)), _) => break,
                    (Ok(other), span) => {
                        bump!(parser);
                        parser.actions.act_on_token((other, span))
                    }
                    (Err(_), _) => unimplemented!(),
                }
            }
            parser.change_context(MacroArgActions::did_parse_macro_arg)
        })
    }
}

impl<'a, I, L, E, R, A, S> Parser<'a, (Result<Token<I, L>, E>, S), R, A>
where
    R: Iterator<Item = (Result<Token<I, L>, E>, S)>,
    A: TokenLineActions<I, L, S>,
    S: Clone,
{
    fn parse_token_line(mut self) -> NextParser<'a, I, L, E, R, A, S> {
        loop {
            match self.state.token {
                (Ok(Token::Ident(ident)), span) => {
                    bump!(self);
                    match self.actions.act_on_ident(ident, span) {
                        TokenLineRule::TokenSeq(actions) => {
                            self = Parser::from_state(self.state, actions)
                        }
                        TokenLineRule::LineEnd(actions) => {
                            return Parser::from_state(self.state, actions).parse_line_terminator()
                        }
                    }
                }
                (Ok(Token::Sigil(Sigil::Eol)), _) | (Ok(Token::Sigil(Sigil::Eos)), _) => {
                    return self.parse_line_terminator()
                }
                (Ok(token), span) => {
                    bump!(self);
                    self.actions.act_on_token(token, span)
                }
                (Err(_), _) => unimplemented!(),
            }
        }
    }
}

impl<'a, I, L, E, R, A, S> Parser<'a, (Result<Token<I, L>, E>, S), R, A>
where
    R: Iterator<Item = (Result<Token<I, L>, E>, S)>,
    A: Diagnostics<S>,
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
            while !self.token_is_in(terminators) && self.token_kind() != Some(Token::Sigil(Eos)) {
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
        if self.token_kind() == Some(Token::Sigil(Eos)) {
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
    {'a, T, R, A: Diagnostics<S>, S}, Parser<'a, T, R, A>, {actions}, A, S
}

#[cfg(test)]
mod tests {
    use super::mock::IdentKind::*;
    use super::mock::*;
    use super::Token::*;
    use super::*;

    use crate::diag::Message;

    use std::borrow::Borrow;

    #[test]
    fn parse_empty_src() {
        assert_eq_actions(input_tokens![], [instr_line(vec![], 0)])
    }

    #[test]
    fn parse_empty_stmt() {
        assert_eq_actions(
            input_tokens![Eol],
            [instr_line(vec![], 0), instr_line(vec![], 1)],
        )
    }

    fn assert_eq_actions(input: InputTokens, expected: impl Borrow<[TokenStreamAction<MockSpan>]>) {
        let mut parsing_context = TokenStreamActionCollector::new();
        parsing_context = parse_src(with_spans(&input.tokens), parsing_context);
        let mut expected = expected.borrow().to_vec();
        expected.push(input.eos());
        assert_eq!(parsing_context.actions, expected)
    }

    #[test]
    fn parse_nullary_instruction() {
        assert_eq_actions(
            input_tokens![nop @ Ident(BuiltinInstr)],
            [unlabeled(builtin_instr(BuiltinInstr, "nop", []), 1)],
        )
    }

    #[test]
    fn parse_nullary_instruction_after_eol() {
        assert_eq_actions(
            input_tokens![Eol, nop @ Ident(BuiltinInstr)],
            [
                instr_line(vec![], 0),
                unlabeled(builtin_instr(BuiltinInstr, "nop", []), 2),
            ],
        )
    }

    #[test]
    fn parse_nullary_instruction_followed_by_eol() {
        assert_eq_actions(
            input_tokens![daa @ Ident(BuiltinInstr), Eol],
            [
                unlabeled(builtin_instr(BuiltinInstr, "daa", []), 1),
                instr_line(vec![], 2),
            ],
        )
    }

    #[test]
    fn parse_unary_instruction() {
        assert_eq_actions(
            input_tokens![db @ Ident(BuiltinInstr), my_ptr @ Ident(Other)],
            [unlabeled(
                builtin_instr(BuiltinInstr, "db", [expr().ident("my_ptr")]),
                2,
            )],
        )
    }

    #[test]
    fn parse_binary_instruction() {
        assert_eq_actions(
            input_tokens![Ident(BuiltinInstr), Ident(Other), Comma, Literal(())],
            [unlabeled(
                builtin_instr(BuiltinInstr, 0, [expr().ident(1), expr().literal(3)]),
                4,
            )],
        );
    }

    #[test]
    fn parse_two_instructions() {
        let tokens = input_tokens![
            Ident(BuiltinInstr),
            Ident(Other),
            Comma,
            Literal(()),
            Eol,
            ld @ Ident(BuiltinInstr),
            a @ Literal(()),
            Comma,
            some_const @ Ident(Other),
        ];
        let expected = [
            unlabeled(
                builtin_instr(BuiltinInstr, 0, [expr().ident(1), expr().literal(3)]),
                4,
            ),
            unlabeled(
                builtin_instr(
                    BuiltinInstr,
                    "ld",
                    [expr().literal("a"), expr().ident("some_const")],
                ),
                9,
            ),
        ];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_two_instructions_separated_by_blank_line() {
        let tokens = input_tokens![
            Ident(BuiltinInstr),
            Literal(()),
            Comma,
            Ident(Other),
            Eol,
            Eol,
            Ident(BuiltinInstr),
            Ident(Other),
            Comma,
            Literal(()),
        ];
        let expected = [
            unlabeled(
                builtin_instr(BuiltinInstr, 0, [expr().literal(1), expr().ident(3)]),
                4,
            ),
            instr_line(vec![], 5),
            unlabeled(
                builtin_instr(BuiltinInstr, 6, [expr().ident(7), expr().literal(9)]),
                10,
            ),
        ];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_empty_macro_definition() {
        let tokens = input_tokens![Label(Other), Ident(MacroKeyword), Eol, Ident(Endm),];
        let expected_actions = [
            labeled(0, vec![], Some(builtin_instr(MacroKeyword, 1, [])), 2),
            token_line(vec![tokens.ident(3)], 4),
        ];
        assert_eq_actions(tokens, expected_actions);
    }

    #[test]
    fn parse_macro_definition_with_instruction() {
        let tokens = input_tokens![
            Label(Other),
            Ident(MacroKeyword),
            Eol,
            Ident(BuiltinInstr),
            Eol,
            Ident(Endm),
        ];
        let expected_actions = [
            labeled(0, vec![], Some(builtin_instr(MacroKeyword, 1, vec![])), 2),
            token_line(vec![tokens.ident(3)], 4),
            token_line(vec![tokens.ident(5)], 6),
        ];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_nonempty_macro_def_with_two_params() {
        let tokens = input_tokens![
            l @ Label(Other),
            LParen,
            p1 @ Ident(Other),
            Comma,
            p2 @ Ident(Other),
            RParen,
            key @ Ident(MacroKeyword),
            eol @ Eol,
            t1 @ Ident(BuiltinInstr),
            t2 @ Eol,
            endm @ Ident(Endm),
        ];
        let expected = [
            labeled(
                "l",
                ["p1".into(), "p2".into()],
                Some(builtin_instr(MacroKeyword, "key", [])),
                "eol",
            ),
            token_line(vec![tokens.ident("t1")], "t2"),
            token_line(vec![tokens.ident("endm")], 11),
        ];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_label() {
        let tokens = input_tokens![Label(Other), Eol];
        let expected_actions = [labeled(0, vec![], None, 1), instr_line(vec![], 2)];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_two_consecutive_labels() {
        let tokens = input_tokens![Label(Other), Eol, Label(Other)];
        let expected = [labeled(0, vec![], None, 1), labeled(2, vec![], None, 3)];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_labeled_instruction() {
        let tokens = input_tokens![Label(Other), Ident(BuiltinInstr), Eol];
        let expected = [
            labeled(0, vec![], Some(builtin_instr(BuiltinInstr, 1, [])), 2),
            instr_line(vec![], 3),
        ];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_labeled_command_with_eol_separators() {
        let tokens = input_tokens![Label(Other), Eol, Eol, Ident(BuiltinInstr)];
        let expected = [
            labeled(0, vec![], None, 1),
            instr_line(vec![], 2),
            unlabeled(builtin_instr(BuiltinInstr, 3, []), 4),
        ];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_deref_operand() {
        let tokens = input_tokens![
            jp @ Ident(BuiltinInstr),
            open @ LParen,
            hl @ Literal(()),
            close @ RParen,
        ];
        let expected = [unlabeled(
            builtin_instr(
                BuiltinInstr,
                "jp",
                [expr().literal("hl").parentheses("open", "close")],
            ),
            4,
        )];
        assert_eq_actions(tokens, expected)
    }

    #[test]
    fn parse_nullary_macro_call() {
        let tokens = input_tokens![Ident(MacroName)];
        let expected_actions = [unlabeled(macro_instr(0, []), 1)];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_call() {
        let tokens = input_tokens![Ident(MacroName), Literal(())];
        let expected_actions = [unlabeled(macro_instr(0, [tokens.token_seq([1])]), 2)];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_unary_macro_call_with_multiple_terminals() {
        let tokens = input_tokens![Ident(MacroName), Literal(()), Literal(()), Literal(()),];
        let expected_actions = [unlabeled(macro_instr(0, [tokens.token_seq([1, 2, 3])]), 4)];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_binary_macro_call_with_multiple_terminals() {
        let tokens = input_tokens![
            Ident(MacroName),
            Literal(()),
            Literal(()),
            Comma,
            Literal(()),
            Literal(()),
            Literal(()),
        ];
        let expected_actions = [unlabeled(
            macro_instr(0, [tokens.token_seq([1, 2]), tokens.token_seq([4, 5, 6])]),
            7,
        )];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_sum_arg() {
        let tokens = input_tokens![
            Ident(BuiltinInstr),
            x @ Ident(Other),
            plus @ Plus,
            y @ Literal(()),
        ];
        let expected_actions = [unlabeled(
            builtin_instr(
                BuiltinInstr,
                0,
                [expr().ident("x").literal("y").plus("plus")],
            ),
            4,
        )];
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn diagnose_stmt_starting_with_literal() {
        let token: MockSpan = TokenRef::from("a").into();
        assert_eq_actions(
            input_tokens![a @ Literal(())],
            [instr_line(
                vec![InstrLineAction::EmitDiag(
                    Message::UnexpectedToken {
                        token: token.clone(),
                    }
                    .at(token)
                    .into(),
                )],
                1,
            )],
        )
    }

    #[test]
    fn diagnose_missing_comma_in_arg_list() {
        let span: MockSpan = TokenRef::from("unexpected").into();
        assert_eq_actions(
            input_tokens![Ident(BuiltinInstr), Literal(()), unexpected @ Literal(())],
            [unlabeled(
                malformed_builtin_instr(
                    0,
                    [expr().literal(1)],
                    Message::UnexpectedToken {
                        token: span.clone(),
                    }
                    .at(span)
                    .into(),
                ),
                3,
            )],
        )
    }

    #[test]
    fn diagnose_eos_in_param_list() {
        assert_eq_actions(
            input_tokens![label @ Label(Other), LParen, eos @ Eos],
            [instr_line(
                vec![InstrLineAction::Label((
                    (
                        MockIdent(Other, "label".into()),
                        TokenRef::from("label").into(),
                    ),
                    vec![ParamsAction::EmitDiag(arg_error(
                        Message::UnexpectedEof,
                        "eos",
                    ))],
                ))],
                "eos",
            )],
        )
    }

    #[test]
    fn diagnose_unmatched_parentheses() {
        assert_eq_actions(
            input_tokens![Ident(BuiltinInstr), paren @ LParen, Literal(())],
            [unlabeled(
                builtin_instr(
                    BuiltinInstr,
                    0,
                    [expr()
                        .literal(2)
                        .diag(Message::UnmatchedParenthesis, TokenRef::from("paren"))],
                ),
                3,
            )],
        )
    }

    #[test]
    fn recover_from_unexpected_token_in_expr() {
        let paren_span: MockSpan = TokenRef::from("paren").into();
        assert_eq_actions(
            input_tokens![
                Ident(BuiltinInstr),
                paren @ RParen,
                Plus,
                Ident(Other),
                Eol,
                nop @ Ident(BuiltinInstr)
            ],
            [
                unlabeled(
                    builtin_instr(
                        BuiltinInstr,
                        0,
                        [expr().error("paren").diag(
                            Message::UnexpectedToken {
                                token: paren_span.clone(),
                            },
                            paren_span,
                        )],
                    ),
                    4,
                ),
                unlabeled(builtin_instr(BuiltinInstr, "nop", []), 6),
            ],
        )
    }

    #[test]
    fn diagnose_unmatched_parenthesis_at_eol() {
        assert_eq_actions(
            input_tokens![Ident(BuiltinInstr), LParen, Eol],
            [
                unlabeled(
                    builtin_instr(
                        BuiltinInstr,
                        0,
                        [expr()
                            .error(2)
                            .diag(Message::UnmatchedParenthesis, TokenRef::from(1))],
                    ),
                    2,
                ),
                instr_line(vec![], 3),
            ],
        )
    }

    #[test]
    fn diagnose_unexpected_token_in_param_list() {
        let span: MockSpan = TokenRef::from("lit").into();
        assert_eq_actions(
            input_tokens![
                label @ Label(Other),
                LParen,
                lit @ Literal(()),
                RParen,
                key @ Ident(BuiltinInstr),
            ],
            [instr_line(
                vec![
                    InstrLineAction::Label((
                        (
                            MockIdent(Other, "label".into()),
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
                    InstrLineAction::Instr(builtin_instr(BuiltinInstr, "key", [])),
                ],
                5,
            )],
        )
    }

    #[test]
    fn recover_from_other_ident_as_key() {
        let tokens = input_tokens![
            Ident(Other),
            Ident(Other),
            Eol,
            nop @ Ident(BuiltinInstr),
        ];
        let expected = [
            unlabeled(vec![InstrAction::Error(vec![])], 2),
            unlabeled(builtin_instr(BuiltinInstr, "nop", []), 4),
        ];
        assert_eq_actions(tokens, expected)
    }
}
