pub(super) use self::lexer::{LexError, Lexer};

use crate::diag::Diagnostics;
use crate::model::BinOp;

mod lexer;
mod parser;

#[derive(Clone, Debug, PartialEq)]
pub enum Token<I, L> {
    Ident(I),
    Label(I),
    Literal(L),
    Simple(SimpleToken),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SimpleToken {
    Comma,
    Dot,
    Endm,
    Eof,
    Eol,
    LParen,
    Minus,
    Pipe,
    Plus,
    RParen,
    Slash,
    Star,
}

impl<I, L> From<SimpleToken> for Token<I, L> {
    fn from(simple: SimpleToken) -> Self {
        Token::Simple(simple)
    }
}

pub(super) trait IdentSource {
    type Ident: Clone + PartialEq;
}

pub(super) trait IdentFactory: IdentSource {
    fn mk_ident(&mut self, spelling: &str) -> Self::Ident;
}

#[cfg(test)]
impl<I: Clone + PartialEq, F: for<'a> Fn(&'a str) -> I> IdentSource for F {
    type Ident = I;
}

#[cfg(test)]
impl<I: Clone + PartialEq, F: for<'a> Fn(&'a str) -> I> IdentFactory for F {
    fn mk_ident(&mut self, spelling: &str) -> Self::Ident {
        self(spelling)
    }
}

pub(super) use self::parser::parse_src as parse_token_seq;

// The following traits represent different positions within the grammar's production rules.

// A token stream represents either a tokenized source file or a macro expansion. It is logically
// devided into lines (separated by <Eol> tokens) and ends with an <Eos> token. It has a single
// production rule:
//
//     1. tokem-stream → (line (<Eol> line)*)? <Eos>
//
// A line can be either an instruction line (e.g. a CPU instruction) or a token line (e.g. a line of
// tokens inside a macro definition). Correspondingly, it has two production rules:
//
//     1. line → instr-line
//     2. line → token-line
//
// This parsing ambiguity is resolved according to the semantics of the program so far, thus the
// rule used by the parser is determined by the value returned from
// TokenStreamContext::will_parse_line.
pub(super) trait TokenStreamContext<I, L, S: Clone>: Sized {
    type InstrLineContext: InstrLineContext<I, L, S, ParentContext = Self>;
    type TokenLineContext;

    fn will_parse_line(self) -> LineRule<Self::InstrLineContext, Self::TokenLineContext>;
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
}

// An instruction line begins with an optional label and continues with an optional instruction,
// thus having a single production rule:
//
//     1. instr-line → label? instr?
//
// InstrLineContext::will_parse_label is called by the parser in the following state:
//
//     instr-line → . label? instr?, <Label>
//
// InstrContext as a supertrait handles the states where the label is missing (either parsing an
// instruction or terminating the empty line) whereas InstrLineContext::InstrContext handles the two
// possible states after a label has been successfully parsed. Note that by using two distinct types
// bound by InstrContext we can preven the parser from calling InstrLineContext::will_parse_label
// more than once.
pub(super) trait InstrLineContext<I, L, S: Clone>: InstrContext<I, L, S> {
    type LabelContext: LabelContext<I, S, ParentContext = Self::InstrContext>;
    type InstrContext: InstrContext<I, L, S, ParentContext = Self::ParentContext>;

    fn will_parse_label(self, label: (I, S)) -> Self::LabelContext;
}

// An instruction can be either a builtin instruction (i.e. a CPU instruction or an assembler
// directive) or a macro instruction, previously defined by the program. These two options
// correspond to two production rules:
//
//     1. instr → <Ident> arg-list
//     2. instr → <Ident> macro-arg-list
//
// The ambiguity between these rules gets resolved by InstrContext::will_parse_instr, which performs
// a name lookup to determine whether the identifier is a builtin instruction or a previously
// defined macro. If neither of these cases applies, a third production rule is used:
//
//     3. instr → <Ident> token-seq
//
// The parser uses this rule to recover from an invalid instruction name by throwing away all the
// remaining tokens in the line.
pub(super) trait InstrContext<I, L, S: Clone>: LineEndContext<S> {
    type BuiltinInstrContext: BuiltinInstrContext<
        S,
        Ident = I,
        Literal = L,
        ParentContext = Self::LineEndContext,
    >;
    type MacroDefContext: TokenSeqContext<S, Token = Token<I, L>, Parent = Self::LineEndContext>;
    type MacroCallContext: MacroCallContext<
        S,
        Token = Token<I, L>,
        ParentContext = Self::LineEndContext,
    >;
    type ErrorContext: InstrEndContext<S, ParentContext = Self::LineEndContext>;
    type LineEndContext: LineEndContext<S, ParentContext = Self::ParentContext>;

    fn will_parse_instr(
        self,
        ident: I,
        span: S,
    ) -> InstrRule<
        Self::BuiltinInstrContext,
        Self::MacroCallContext,
        Self::MacroDefContext,
        Self::ErrorContext,
    >;
}

pub(super) trait LineEndContext<S: Clone>: Diagnostics<S> + Sized {
    type ParentContext;

    fn did_parse_line(self) -> Self::ParentContext;
}

pub(super) trait InstrEndContext<S: Clone>: Diagnostics<S> + Sized {
    type ParentContext;

    fn did_parse_instr(self) -> Self::ParentContext;
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum InstrRule<C, M, D, E> {
    BuiltinInstr(C),
    MacroInstr(M),
    MacroDef(D),
    Error(E),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Keyword {
    Macro,
}

#[cfg(test)]
impl<C, M, D, E> InstrRule<C, M, D, E> {
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

    pub fn macro_def(self) -> Option<D> {
        match self {
            InstrRule::MacroDef(context) => Some(context),
            _ => None,
        }
    }

    pub fn error(self) -> Option<E> {
        match self {
            InstrRule::Error(context) => Some(context),
            _ => None,
        }
    }
}

pub(super) trait BuiltinInstrContext<S: Clone>: InstrEndContext<S> {
    type Ident;
    type Literal;
    type ArgContext: ExprContext<S, Ident = Self::Ident, Literal = Self::Literal>
        + FinalContext<ReturnTo = Self>;

    fn add_argument(self) -> Self::ArgContext;
}

pub(super) trait FinalContext {
    type ReturnTo;

    fn exit(self) -> Self::ReturnTo;
}

pub(super) trait ExprContext<S: Clone>: Diagnostics<S> {
    type Ident;
    type Literal;

    fn push_atom(&mut self, atom: (ExprAtom<Self::Ident, Self::Literal>, S));
    fn apply_operator(&mut self, operator: (Operator, S));
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprAtom<I, L> {
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

pub(super) trait LabelContext<I, S: Clone>: Diagnostics<S> {
    type ParentContext;

    fn act_on_param(&mut self, param: (I, S));
    fn did_parse_label(self) -> Self::ParentContext;
}

pub(super) trait MacroCallContext<S: Clone>: InstrEndContext<S> {
    type Token;
    type MacroArgContext: TokenSeqContext<S, Token = Self::Token, Parent = Self>;

    fn enter_macro_arg(self) -> Self::MacroArgContext;
}

pub(super) trait TokenSeqContext<S: Clone>: Diagnostics<S> {
    type Token;
    type Parent;
    fn push_token(&mut self, token: (Self::Token, S));
    fn exit(self) -> Self::Parent;
}
