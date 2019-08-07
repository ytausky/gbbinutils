pub(super) use self::lexer::{LexError, Lexer};
pub(super) use self::parser::ParseTokenStream;

use self::parser::DefaultParser;

use crate::diag::Diagnostics;
use crate::model::BinOp;

#[cfg(test)]
pub(super) use self::mock::*;

mod lexer;
mod parser;

pub(super) trait ParserFactory<I, L, E, S: Clone> {
    type Parser: ParseTokenStream<I, L, E, S>;

    fn mk_parser(&mut self) -> Self::Parser;
}

pub(super) struct DefaultParserFactory;

impl<I, L, E, S: Clone> ParserFactory<I, L, E, S> for DefaultParserFactory {
    type Parser = DefaultParser;

    fn mk_parser(&mut self) -> Self::Parser {
        DefaultParser
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Token<I, L> {
    Ident(I),
    Label(I),
    Literal(L),
    Sigil(Sigil),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sigil {
    Comma,
    Dot,
    Eos,
    Eol,
    LParen,
    Minus,
    Pipe,
    Plus,
    RParen,
    Slash,
    Star,
}

impl<I, L> From<Sigil> for Token<I, L> {
    fn from(sigil: Sigil) -> Self {
        Token::Sigil(sigil)
    }
}

pub(super) trait IdentSource {
    type Ident: Clone + PartialEq + AsRef<str>;
}

pub(super) trait IdentFactory: IdentSource {
    fn mk_ident(&mut self, spelling: &str) -> Self::Ident;
}

#[cfg(test)]
impl<I: Clone + PartialEq + AsRef<str>, F: for<'a> Fn(&'a str) -> I> IdentSource for F {
    type Ident = I;
}

#[cfg(test)]
impl<I: Clone + PartialEq + AsRef<str>, F: for<'a> Fn(&'a str) -> I> IdentFactory for F {
    fn mk_ident(&mut self, spelling: &str) -> Self::Ident {
        self(spelling)
    }
}

// The following traits represent different positions within the grammar's production rules.

// A token stream represents either a tokenized source file or a macro expansion. It is logically
// divided into lines (separated by <Eol> tokens) and ends with an <Eos> token. It has a single
// production rule:
//
//     1. token-stream → (line (<Eol> line)*)? <Eos>
//
// A line can be either an instruction line (e.g. a CPU instruction) or a token line (e.g. a line of
// tokens inside a macro definition). Correspondingly, it has two production rules:
//
//     1. line → instr-line
//     2. line → token-line
//
// This parsing ambiguity is resolved according to the semantics of the program so far, thus the
// rule used by the parser is determined by the value returned from
// TokenStreamActions::will_parse_line.
pub(super) trait TokenStreamActions<I, L, S: Clone>: Sized {
    type InstrLineActions: InstrLineActions<I, L, S, Next = Self>;
    type TokenLineActions: TokenLineActions<
        I,
        L,
        S,
        ContextFinalizer = Self::TokenLineFinalizer,
        Next = Self,
    >;
    type TokenLineFinalizer: LineFinalizer<S, Next = Self>;

    fn will_parse_line(self) -> LineRule<Self::InstrLineActions, Self::TokenLineActions>;
    fn act_on_eos(self, span: S) -> Self;
}

pub(super) enum LineRule<I, T> {
    InstrLine(I),
    TokenLine(T),
}

#[cfg(test)]
impl<I, T> LineRule<I, T> {
    pub fn into_instr_line(self) -> I {
        match self {
            LineRule::InstrLine(context) => context,
            _ => panic!("expected instruction line"),
        }
    }

    pub fn into_token_line(self) -> T {
        match self {
            LineRule::TokenLine(context) => context,
            _ => panic!("expected token line"),
        }
    }
}

// An instruction line begins with an optional label and continues with an optional instruction,
// thus having a single production rule:
//
//     1. instr-line → label? instr?
//
// InstrLineActions::will_parse_label is called by the parser in the following state:
//
//     instr-line → . label? instr?, <Label>
//
// InstrActions as a supertrait handles the states where the label is missing (either parsing an
// instruction or terminating the empty line) whereas InstrLineActions::InstrActions handles the two
// possible states after a label has been successfully parsed. Note that by using two distinct types
// bound by InstrActions we can prevent the parser from calling InstrLineActions::will_parse_label
// more than once on the same line.
pub(super) trait InstrLineActions<I, L, S: Clone>: InstrActions<I, L, S> {
    type LabelActions: LabelActions<I, S, Next = Self::InstrActions>;
    type InstrActions: InstrActions<I, L, S, Next = Self::Next>;

    fn will_parse_label(self, label: (I, S)) -> Self::LabelActions;
}

// An instruction can be either a builtin instruction (i.e. a CPU instruction or an assembler
// directive) or a macro instruction previously defined by the program. These two options correspond
// to two production rules:
//
//     1. instr → builtin-instr
//     2. instr → macro-instr
//
// The ambiguity between these rules gets resolved by InstrActions::will_parse_instr, which performs
// a name lookup to determine whether the identifier is a builtin instruction or a previously
// defined macro. If neither of these cases applies, a third production rule is used:
//
//     3. instr → <Ident> token-seq
//
// The parser uses this rule to recover from an invalid instruction name by throwing away all the
// remaining tokens in the line.
pub(super) trait InstrActions<I, L, S: Clone>: LineFinalizer<S> {
    type BuiltinInstrActions: BuiltinInstrActions<I, L, S, Next = Self::LineFinalizer>;
    type MacroInstrActions: MacroInstrActions<S, Token = Token<I, L>, Next = Self::LineFinalizer>;
    type ErrorActions: InstrFinalizer<S, Next = Self::LineFinalizer>;
    type LineFinalizer: LineFinalizer<S, Next = Self::Next>;

    fn will_parse_instr(
        self,
        ident: I,
        span: S,
    ) -> InstrRule<Self::BuiltinInstrActions, Self::MacroInstrActions, Self::ErrorActions>;
}

pub(super) trait LineFinalizer<S: Clone>: Diagnostics<S> + Sized {
    type Next;

    fn did_parse_line(self, span: S) -> Self::Next;
}

pub(super) trait InstrFinalizer<S: Clone>: Diagnostics<S> + Sized {
    type Next;

    fn did_parse_instr(self) -> Self::Next;
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum InstrRule<C, M, E> {
    BuiltinInstr(C),
    MacroInstr(M),
    Error(E),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Keyword {}

#[cfg(test)]
impl<C, M, E> InstrRule<C, M, E> {
    pub fn into_builtin_instr(self) -> C {
        match self {
            InstrRule::BuiltinInstr(context) => context,
            _ => panic!("expected builtin instruction context"),
        }
    }

    pub fn into_macro_instr(self) -> M {
        match self {
            InstrRule::MacroInstr(context) => context,
            _ => panic!("expected macro instruction"),
        }
    }

    pub fn error(self) -> Option<E> {
        match self {
            InstrRule::Error(context) => Some(context),
            _ => None,
        }
    }
}

// Builtin instructions have a single production rule:
//
//     1. builtin-instr → <Ident> (arg (<Comma> arg)*)?
//
// BuiltinInstrActions represents any position in this rule after the initial <Ident>.
pub(super) trait BuiltinInstrActions<I, L, S: Clone>: InstrFinalizer<S> {
    type ArgActions: ArgActions<I, L, S> + ArgFinalizer<Next = Self>;

    fn will_parse_arg(self) -> Self::ArgActions;
}

pub(super) trait ArgFinalizer {
    type Next;

    fn did_parse_arg(self) -> Self::Next;
}

// An argument is a recursive expression with the following production rules:
//
//     1. arg → <Ident>
//     2. arg → <Literal>
//     3. arg → <Dot>
//     4. arg → arg <LParen> (arg (<Comma> arg)*)? <RParen>
//     5. arg → arg <Star> arg
//     6. ...
//
// To handle precedence and associativity, the parser uses a reverse Polish notation protocol.
pub(super) trait ArgActions<I, L, S: Clone>: Diagnostics<S> {
    fn act_on_atom(&mut self, atom: ExprAtom<I, L>, span: S);
    fn act_on_operator(&mut self, operator: Operator, span: S);
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprAtom<I, L> {
    Error,
    Ident(I),
    Literal(L),
    LocationCounter,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operator {
    Unary(UnaryOperator),
    Binary(BinOp),
    FnCall(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub enum UnaryOperator {
    Parentheses,
}

pub(super) trait LabelActions<I, S: Clone>: Diagnostics<S> {
    type Next;

    fn act_on_param(&mut self, param: I, span: S);
    fn did_parse_label(self) -> Self::Next;
}

pub(super) trait MacroInstrActions<S: Clone>: InstrFinalizer<S> {
    type Token;
    type MacroArgActions: MacroArgActions<S, Token = Self::Token, Next = Self>;

    fn will_parse_macro_arg(self) -> Self::MacroArgActions;
}

pub(super) trait MacroArgActions<S: Clone>: Diagnostics<S> {
    type Token;
    type Next;

    fn act_on_token(&mut self, token: (Self::Token, S));
    fn did_parse_macro_arg(self) -> Self::Next;
}

pub(super) trait TokenLineActions<I, L, S: Clone>: LineFinalizer<S> {
    type ContextFinalizer: LineFinalizer<S, Next = Self::Next>;

    fn act_on_token(&mut self, token: Token<I, L>, span: S);
    fn act_on_ident(self, ident: I, span: S) -> TokenLineRule<Self, Self::ContextFinalizer>;
}

pub(super) enum TokenLineRule<T, E> {
    TokenSeq(T),
    LineEnd(E),
}

#[cfg(test)]
impl<T, E> TokenLineRule<T, E> {
    pub fn into_line_end(self) -> E {
        match self {
            TokenLineRule::LineEnd(context) => context,
            _ => panic!("expected token sequence"),
        }
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::log::Log;

    pub(in crate::analysis) struct MockParserFactory<T> {
        log: Log<T>,
    }

    impl<T> MockParserFactory<T> {
        pub fn new(log: Log<T>) -> Self {
            Self { log }
        }
    }

    impl<I, L, E, T, S: Clone> ParserFactory<I, L, E, S> for MockParserFactory<T>
    where
        T: From<ParserEvent<I, L, E, S>>,
    {
        type Parser = MockParser<T>;

        fn mk_parser(&mut self) -> Self::Parser {
            MockParser {
                log: self.log.clone(),
            }
        }
    }

    pub(in crate::analysis) struct MockParser<T> {
        log: Log<T>,
    }

    impl<I, L, E, T, S: Clone> ParseTokenStream<I, L, E, S> for MockParser<T>
    where
        T: From<ParserEvent<I, L, E, S>>,
    {
        fn parse_token_stream<R, A>(&mut self, tokens: R, actions: A) -> A
        where
            R: IntoIterator<Item = (Result<Token<I, L>, E>, S)>,
            A: TokenStreamActions<I, L, S>,
        {
            self.log
                .push(ParserEvent::ParseTokenStream(tokens.into_iter().collect()));
            actions
        }
    }

    #[derive(Debug, PartialEq)]
    pub(in crate::analysis) enum ParserEvent<I, L, E, S> {
        ParseTokenStream(Vec<TokenStreamItem<I, L, E, S>>),
    }

    type TokenStreamItem<I, L, E, S> = (Result<Token<I, L>, E>, S);
}
