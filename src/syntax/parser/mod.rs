use super::actions::*;
use super::Sigil::*;
use super::{Sigil, Token};

use crate::session::diagnostics::Message;

macro_rules! bump {
    ($parser:expr) => {
        $parser.state.token = $parser.actions.next_token().unwrap()
    };
}

#[cfg(test)]
macro_rules! input_tokens {
    ($($tokens:tt)*) => {{
        let mut input = InputTokens {
            tokens: Vec::new(),
            names: std::collections::HashMap::new(),
        };
        input_tokens_impl!(input, $($tokens)*);
        if input
            .tokens
            .last()
            .map(|(token, _)| *token != Eos.into())
            .unwrap_or(true)
        {
            let eos_id = input.tokens.len().into();
            input.tokens.push((Eos.into(), eos_id))
        }
        input
    }};
}

#[cfg(test)]
macro_rules! add_token {
    ($input:expr, $token:expr) => {
        let id = $input.tokens.len();
        $input.insert_token(id, $token.into())
    };
    ($input:expr, $name:ident @ $token:expr) => {
        let id = stringify!($name);
        $input.names.insert(id.into(), $input.tokens.len());
        $input.insert_token(id, $token.into())
    };
}

#[cfg(test)]
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

mod expr;

pub(crate) trait ParserFactory<I, L, E, S: Clone> {
    type Parser: ParseTokenStream<I, L, E, S>;

    fn mk_parser(&mut self) -> Self::Parser;
}

pub(crate) trait ParseTokenStream<I, L, E, S: Clone> {
    fn parse_token_stream<A>(&mut self, actions: A) -> A
    where
        A: TokenStreamContext<Ident = I, Literal = L, Error = E, Span = S>;
}

pub struct DefaultParserFactory;

impl<I, L, E, S: Clone> ParserFactory<I, L, E, S> for DefaultParserFactory {
    type Parser = DefaultParser;

    fn mk_parser(&mut self) -> Self::Parser {
        DefaultParser
    }
}

pub struct DefaultParser;

impl<I, L, E, S: Clone> ParseTokenStream<I, L, E, S> for DefaultParser {
    fn parse_token_stream<A>(&mut self, actions: A) -> A
    where
        A: TokenStreamContext<Ident = I, Literal = L, Error = E, Span = S>,
        S: Clone,
    {
        let Parser {
            state: ParserState { token, .. },
            actions,
            ..
        } = Parser::new(actions).parse_token_stream();
        assert_eq!(
            token.0.ok().as_ref().map(Token::kind),
            Some(Token::Sigil(Sigil::Eos))
        );
        actions.act_on_eos(token.1)
    }
}

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

struct Parser<T, A> {
    state: ParserState<T>,
    actions: A,
}

struct ParserState<T> {
    token: T,
    parsed_eos: bool,
    recovery: Option<RecoveryState>,
}

enum RecoveryState {
    DiagnosedEof,
}

impl<A: ParsingContext> Parser<LexerOutput<A::Ident, A::Literal, A::Error, A::Span>, A> {
    fn new(mut actions: A) -> Self {
        Self::from_state(
            ParserState {
                token: actions.next_token().unwrap(),
                parsed_eos: false,
                recovery: None,
            },
            actions,
        )
    }
}

impl<T, A> Parser<T, A> {
    fn from_state(state: ParserState<T>, actions: A) -> Self {
        Parser { state, actions }
    }

    fn change_context<D, F: FnOnce(A) -> D>(self, f: F) -> Parser<T, D> {
        Parser::from_state(self.state, f(self.actions))
    }
}

impl<I, L, E, S, A> Parser<(Result<Token<I, L>, E>, S), A> {
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

impl<I, L, E, C, S> Parser<(Result<Token<I, L>, E>, S), C>
where
    C: TokenStreamContext<Ident = I, Literal = L, Error = E, Span = S>,
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

type NextParser<I, L, E, A, S> = Parser<(Result<Token<I, L>, E>, S), <A as LineFinalizer>::Next>;

impl<I, L, E, A, S> Parser<(Result<Token<I, L>, E>, S), A>
where
    A: InstrLineContext<Ident = I, Literal = L, Error = E, Span = S>,
    S: Clone,
{
    fn parse_instr_line(mut self) -> NextParser<I, L, E, A, S> {
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
                        .change_context(LabelContext::did_parse_label)
                        .parse_line_terminator();
                }
            }
            parser
                .change_context(LabelContext::did_parse_label)
                .parse_unlabeled_stmt()
        } else {
            self.parse_unlabeled_stmt()
        }
    }
}

impl<I, L, E, A, S> Parser<(Result<Token<I, L>, E>, S), A>
where
    A: InstrContext<Ident = I, Literal = L, Error = E, Span = S>,
    S: Clone,
{
    fn parse_unlabeled_stmt(mut self) -> NextParser<I, L, E, A, S> {
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
                let stripped = self.actions.strip_span(&span);
                self.actions
                    .emit_diag(Message::UnexpectedToken { token: stripped }.at(span));
                self.parse_line_terminator()
            }
        }
    }

    fn parse_key(self, key: I, span: S) -> NextParser<I, L, E, A, S> {
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

impl<I, L, E, A, S> Parser<(Result<Token<I, L>, E>, S), A>
where
    A: LineFinalizer<Ident = I, Literal = L, Error = E, Span = S>,
    S: Clone,
{
    fn parse_line_terminator(mut self) -> NextParser<I, L, E, A, S> {
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

impl<I, L, E, A, S> Parser<(Result<Token<I, L>, E>, S), A>
where
    A: BuiltinInstrContext<Ident = I, Literal = L, Error = E, Span = S>,
    S: Clone,
{
    fn parse_argument_list(self) -> Self {
        self.parse_terminated_list(Comma.into(), LINE_FOLLOW_SET, Parser::parse_argument)
    }

    fn parse_argument(self) -> Self {
        self.change_context(BuiltinInstrContext::will_parse_arg)
            .parse()
            .change_context(ArgFinalizer::did_parse_arg)
    }
}

impl<I, L, E, A, S> Parser<(Result<Token<I, L>, E>, S), A>
where
    A: LabelContext<Ident = I, Literal = L, Error = E, Span = S>,
    S: Clone,
{
    fn parse_param(mut self) -> Self {
        match self.state.token.0 {
            Ok(Token::Ident(ident)) => {
                self.actions.act_on_param(ident, self.state.token.1);
                bump!(self)
            }
            _ => self = self.diagnose_unexpected_token(),
        };
        self
    }
}

impl<I, L, E, A, S> Parser<(Result<Token<I, L>, E>, S), A>
where
    A: MacroInstrContext<Ident = I, Literal = L, Error = E, Span = S>,
    S: Clone,
{
    fn parse_macro_call(self) -> Self {
        self.parse_macro_arg_list()
    }

    fn parse_macro_arg_list(self) -> Self {
        self.parse_terminated_list(Comma.into(), LINE_FOLLOW_SET, |p| {
            let mut parser = p.change_context(MacroInstrContext::will_parse_macro_arg);
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
            parser.change_context(MacroArgContext::did_parse_macro_arg)
        })
    }
}

impl<I, L, E, A, S> Parser<(Result<Token<I, L>, E>, S), A>
where
    A: TokenLineContext<Ident = I, Literal = L, Error = E, Span = S>,
    S: Clone,
{
    fn parse_token_line(mut self) -> NextParser<I, L, E, A, S> {
        loop {
            match self.state.token {
                (Ok(Token::Ident(ident)), span) => {
                    bump!(self);
                    match self.actions.act_on_mnemonic(ident, span) {
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

impl<I, L, E, A, S> Parser<(Result<Token<I, L>, E>, S), A>
where
    A: ParsingContext<Ident = I, Literal = L, Error = E, Span = S>,
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
                self.actions
                    .emit_diag(Message::UnexpectedEof.at(self.state.token.1.clone()));
                self.state.recovery = Some(RecoveryState::DiagnosedEof)
            }
        } else {
            let token = self.state.token.1;
            bump!(self);
            let stripped = self.actions.strip_span(&token);
            self.actions
                .emit_diag(Message::UnexpectedToken { token: stripped }.at(token))
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::Token::*;
    use super::*;

    use crate::expr::BinOp;
    use crate::session::diagnostics::{CompactDiag, Merge, Message};
    use crate::syntax::actions::mock::IdentKind::*;
    use crate::syntax::actions::mock::*;

    use std::borrow::Borrow;
    use std::collections::HashMap;

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

    fn assert_eq_actions(
        input: InputTokens,
        expected: impl Borrow<[TokenStreamAction<MockIdent, MockLiteral, MockSpan>]>,
    ) {
        let tokens = &mut with_spans(&input.tokens);
        let mut parsing_context = TokenStreamActionCollector::new((), tokens, MockIdent::annotate);
        parsing_context = DefaultParser.parse_token_stream(parsing_context);
        let mut expected = expected.borrow().to_vec();
        expected.push(input.eos());
        assert_eq!(parsing_context.into_actions(), expected)
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

    impl MockIdent {
        pub fn annotate(&self) -> IdentKind {
            self.0
        }
    }

    pub(super) type MockSpan = crate::session::diagnostics::MockSpan<TokenRef>;

    pub(super) fn with_spans<'a>(
        tokens: impl IntoIterator<Item = &'a (MockToken, TokenRef)>,
    ) -> impl Iterator<Item = (Result<MockToken, ()>, MockSpan)> {
        tokens.into_iter().cloned().map(|(t, r)| (Ok(t), r.into()))
    }

    #[derive(Debug, PartialEq)]
    pub struct MacroId(pub TokenRef);

    pub(super) fn expr() -> SymExpr {
        SymExpr(Vec::new())
    }

    impl SymExpr {
        pub fn ident(self, token: impl Into<TokenRef>) -> Self {
            self.push(token, |t| ExprAtom::Ident(MockIdent(IdentKind::Other, t)))
        }

        pub fn literal(self, token: impl Into<TokenRef>) -> Self {
            self.push(token, |t| ExprAtom::Literal(MockLiteral(t)))
        }

        pub fn location_counter(self, token: impl Into<TokenRef>) -> Self {
            self.push(token, |_| ExprAtom::LocationCounter)
        }

        fn push(
            mut self,
            token: impl Into<TokenRef>,
            atom_ctor: impl Fn(TokenRef) -> ExprAtom<MockIdent, MockLiteral>,
        ) -> Self {
            let token_ref = token.into();
            self.0.push(ExprAction::PushAtom(
                atom_ctor(token_ref.clone()),
                token_ref.into(),
            ));
            self
        }

        pub fn equals(mut self, token: impl Into<TokenRef>) -> Self {
            self.0.push(ExprAction::ApplyOperator(
                Operator::Binary(BinOp::Equality),
                token.into().into(),
            ));
            self
        }

        pub fn divide(mut self, token: impl Into<TokenRef>) -> Self {
            self.0.push(ExprAction::ApplyOperator(
                Operator::Binary(BinOp::Division),
                token.into().into(),
            ));
            self
        }

        pub fn multiply(mut self, token: impl Into<TokenRef>) -> Self {
            self.0.push(ExprAction::ApplyOperator(
                Operator::Binary(BinOp::Multiplication),
                token.into().into(),
            ));
            self
        }

        pub fn parentheses(
            mut self,
            left: impl Into<TokenRef>,
            right: impl Into<TokenRef>,
        ) -> Self {
            let span = MockSpan::merge(left.into(), right.into());
            self.0.push(ExprAction::ApplyOperator(
                Operator::Unary(UnaryOperator::Parentheses),
                span,
            ));
            self
        }

        pub fn plus(mut self, token: impl Into<TokenRef>) -> Self {
            self.0.push(ExprAction::ApplyOperator(
                Operator::Binary(BinOp::Plus),
                token.into().into(),
            ));
            self
        }

        pub fn minus(mut self, token: impl Into<TokenRef>) -> Self {
            self.0.push(ExprAction::ApplyOperator(
                Operator::Binary(BinOp::Minus),
                token.into().into(),
            ));
            self
        }

        pub fn bit_or(mut self, token: impl Into<TokenRef>) -> Self {
            self.0.push(ExprAction::ApplyOperator(
                Operator::Binary(BinOp::BitOr),
                token.into().into(),
            ));
            self
        }

        pub fn fn_call(mut self, args: usize, span: impl Into<MockSpan>) -> Self {
            self.0.push(ExprAction::ApplyOperator(
                Operator::FnCall(args),
                span.into(),
            ));
            self
        }

        pub fn error(mut self, span: impl Into<TokenRef>) -> Self {
            self.0
                .push(ExprAction::PushAtom(ExprAtom::Error, span.into().into()));
            self
        }

        pub fn diag(mut self, message: Message<MockSpan>, highlight: impl Into<MockSpan>) -> Self {
            self.0
                .push(ExprAction::EmitDiag(message.at(highlight.into()).into()));
            self
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    pub struct SymCommand(pub TokenRef);

    #[derive(Clone, Debug, PartialEq)]
    pub struct MockIdent(pub IdentKind, pub TokenRef);

    #[derive(Clone, Debug, PartialEq)]
    pub struct MockLiteral(pub TokenRef);

    pub type MockToken = Token<MockIdent, MockLiteral>;

    fn mk_mock_token(
        id: impl Into<TokenRef>,
        token: Token<IdentKind, ()>,
    ) -> (MockToken, TokenRef) {
        let token_ref = id.into();
        (
            match token {
                Ident(kind) => Ident(MockIdent(kind, token_ref.clone())),
                Label(kind) => Label(MockIdent(kind, token_ref.clone())),
                Literal(()) => Literal(MockLiteral(token_ref.clone())),
                Sigil(sigil) => Sigil(sigil),
            },
            token_ref,
        )
    }

    pub(super) struct InputTokens {
        pub tokens: Vec<(MockToken, TokenRef)>,
        pub names: HashMap<String, usize>,
    }

    impl InputTokens {
        pub fn insert_token(&mut self, id: impl Into<TokenRef>, token: Token<IdentKind, ()>) {
            self.tokens.push(mk_mock_token(id, token))
        }

        pub fn token_seq<T>(
            &self,
            tokens: impl Borrow<[T]>,
        ) -> Vec<TokenSeqAction<MockIdent, MockLiteral, MockSpan>>
        where
            T: Clone + Into<TokenRef>,
        {
            tokens
                .borrow()
                .iter()
                .cloned()
                .map(Into::into)
                .map(|t| TokenSeqAction::PushToken(self.token(t)))
                .collect()
        }

        pub fn token(&self, token_ref: impl Into<TokenRef>) -> (MockToken, MockSpan) {
            let token_ref = token_ref.into();
            let id = match &token_ref {
                TokenRef::Id(n) => *n,
                TokenRef::Name(name) => self.names[name],
            };
            (self.tokens[id].0.clone(), token_ref.into())
        }

        pub fn ident(
            &self,
            token_ref: impl Into<TokenRef>,
        ) -> TokenLineAction<MockIdent, MockLiteral, MockSpan> {
            match self.token(token_ref.into()) {
                (Token::Ident(ident), span) => TokenLineAction::Ident((ident, span)),
                _ => panic!("expected identifier"),
            }
        }

        pub fn eos(&self) -> TokenStreamAction<MockIdent, MockLiteral, MockSpan> {
            TokenStreamAction::Eos(self.tokens.last().unwrap().1.clone().into())
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    pub enum TokenRef {
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

    #[derive(Clone)]
    pub(super) struct SymExpr(pub Vec<ExprAction<MockIdent, MockLiteral, MockSpan>>);

    pub(super) fn instr_line(
        actions: Vec<InstrLineAction<MockIdent, MockLiteral, MockSpan>>,
        terminator: impl Into<TokenRef>,
    ) -> TokenStreamAction<MockIdent, MockLiteral, MockSpan> {
        TokenStreamAction::InstrLine(actions, terminator.into().into())
    }

    pub(super) fn token_line(
        actions: Vec<TokenLineAction<MockIdent, MockLiteral, MockSpan>>,
        terminator: impl Into<TokenRef>,
    ) -> TokenStreamAction<MockIdent, MockLiteral, MockSpan> {
        TokenStreamAction::TokenLine(actions, terminator.into().into())
    }

    pub(super) fn labeled(
        label: impl Into<TokenRef>,
        params: impl Borrow<[TokenRef]>,
        actions: Option<Vec<InstrAction<MockIdent, MockLiteral, MockSpan>>>,
        terminator: impl Into<TokenRef>,
    ) -> TokenStreamAction<MockIdent, MockLiteral, MockSpan> {
        let label = label.into();
        let mut instr_line_actions = vec![InstrLineAction::Label((
            (MockIdent(IdentKind::Other, label.clone()), label.into()),
            convert_params(params),
        ))];
        if let Some(actions) = actions {
            instr_line_actions.push(InstrLineAction::Instr(actions))
        }
        TokenStreamAction::InstrLine(instr_line_actions, terminator.into().into())
    }

    pub(super) fn unlabeled(
        actions: Vec<InstrAction<MockIdent, MockLiteral, MockSpan>>,
        terminator: impl Into<TokenRef>,
    ) -> TokenStreamAction<MockIdent, MockLiteral, MockSpan> {
        TokenStreamAction::InstrLine(
            vec![InstrLineAction::Instr(actions)],
            terminator.into().into(),
        )
    }

    pub(super) fn builtin_instr(
        kind: IdentKind,
        id: impl Into<TokenRef>,
        args: impl Borrow<[SymExpr]>,
    ) -> Vec<InstrAction<MockIdent, MockLiteral, MockSpan>> {
        let id = id.into();
        vec![InstrAction::BuiltinInstr {
            builtin_instr: (MockIdent(kind, id.clone()), id.into()),
            actions: args
                .borrow()
                .iter()
                .cloned()
                .map(|SymExpr(expr)| BuiltinInstrAction::AddArgument { actions: expr })
                .collect(),
        }]
    }

    pub(super) fn malformed_builtin_instr(
        id: impl Into<TokenRef>,
        args: impl Borrow<[SymExpr]>,
        diag: CompactDiag<MockSpan>,
    ) -> Vec<InstrAction<MockIdent, MockLiteral, MockSpan>> {
        let id = id.into();
        vec![InstrAction::BuiltinInstr {
            builtin_instr: (MockIdent(IdentKind::BuiltinInstr, id.clone()), id.into()),
            actions: args
                .borrow()
                .iter()
                .cloned()
                .map(|SymExpr(expr)| BuiltinInstrAction::AddArgument { actions: expr })
                .chain(std::iter::once(BuiltinInstrAction::EmitDiag(diag)))
                .collect(),
        }]
    }

    pub(super) fn macro_instr(
        id: impl Into<TokenRef>,
        args: impl Borrow<[Vec<TokenSeqAction<MockIdent, MockLiteral, MockSpan>>]>,
    ) -> Vec<InstrAction<MockIdent, MockLiteral, MockSpan>> {
        let id = id.into();
        vec![InstrAction::MacroInstr {
            name: (MockIdent(IdentKind::MacroName, id.clone()), id.into()),
            actions: args
                .borrow()
                .iter()
                .cloned()
                .map(MacroInstrAction::MacroArg)
                .collect(),
        }]
    }

    fn convert_params(params: impl Borrow<[TokenRef]>) -> Vec<ParamsAction<MockIdent, MockSpan>> {
        params
            .borrow()
            .iter()
            .cloned()
            .map(|t| ParamsAction::AddParameter(MockIdent(IdentKind::Other, t.clone()), t.into()))
            .collect()
    }

    pub(super) fn arg_error(
        message: Message<MockSpan>,
        highlight: impl Into<TokenRef>,
    ) -> CompactDiag<MockSpan> {
        message.at(highlight.into().into()).into()
    }

    #[test]
    fn test_token_macro() {
        let tokens = input_tokens![
            my_tok @ Plus,
            Literal(()),
            next_one @ Star,
        ];
        assert_eq!(
            tokens.tokens,
            [
                (Plus.into(), "my_tok".into()),
                (Literal(MockLiteral(1.into())), 1.into()),
                (Star.into(), "next_one".into()),
                (Eos.into(), 3.into()),
            ]
        );
        assert_eq!(tokens.names.get("my_tok"), Some(&0));
        assert_eq!(tokens.names.get("next_one"), Some(&2))
    }
}
