pub mod keyword;
pub mod lexer;
mod parser;

pub fn tokenize(src: &str) -> self::lexer::Lexer {
    self::lexer::Lexer::new(src)
}

pub fn parse_token_seq<I, BC>(tokens: I, actions: BC)
where
    I: Iterator<Item = Token<BC::TokenSpec>>,
    BC: BlockContext,
{
    self::parser::parse_src(tokens, actions)
}

#[derive(Clone, Debug, PartialEq)]
pub enum Token<S: TokenSpec> {
    Atom(S::Atom),
    ClosingBracket,
    Colon,
    Comma,
    Command(S::Command),
    Endm,
    Eol,
    Label(S::Label),
    Macro,
    OpeningBracket,
}

pub trait TokenSpec {
    type Atom;
    type Command;
    type Label;
}

impl<T: AsRef<str>> TokenSpec for T {
    type Atom = Atom<T>;
    type Command = keyword::Command;
    type Label = T;
}

#[derive(Clone, Debug, PartialEq)]
pub enum Atom<S> {
    Ident(S),
    Operand(keyword::Operand),
    Number(i32),
    String(S),
}

pub trait BlockContext
where
    Self: Sized,
{
    type TokenSpec: TokenSpec;
    type CommandContext: CommandContext<Terminal = Token<Self::TokenSpec>, EnclosingContext = Self>;
    type MacroDefContext: TerminalSeqContext<
        Terminal = Token<Self::TokenSpec>,
        EnclosingContext = Self,
    >;
    type MacroInvocationContext: MacroInvocationContext<
        Terminal = Token<Self::TokenSpec>,
        EnclosingContext = Self,
    >;
    fn add_label(&mut self, label: <Self::TokenSpec as TokenSpec>::Label);
    fn enter_command(self, name: <Self::TokenSpec as TokenSpec>::Command) -> Self::CommandContext;
    fn enter_macro_def(self, name: <Self::TokenSpec as TokenSpec>::Label) -> Self::MacroDefContext;
    fn enter_macro_invocation(
        self,
        name: <Self::TokenSpec as TokenSpec>::Atom,
    ) -> Self::MacroInvocationContext;
}

pub trait CommandContext {
    type Terminal;
    type EnclosingContext;
    fn add_argument(&mut self, expr: SynExpr<Self::Terminal>);
    fn exit_command(self) -> Self::EnclosingContext;
}

pub trait MacroInvocationContext
where
    Self: Sized,
{
    type Terminal;
    type EnclosingContext;
    type MacroArgContext: TerminalSeqContext<Terminal = Self::Terminal, EnclosingContext = Self>;
    fn enter_macro_arg(self) -> Self::MacroArgContext;
    fn exit_macro_invocation(self) -> Self::EnclosingContext;
}

pub trait TerminalSeqContext {
    type Terminal;
    type EnclosingContext;
    fn push_terminal(&mut self, terminal: Self::Terminal);
    fn exit_terminal_seq(self) -> Self::EnclosingContext;
}

#[derive(Clone, Debug, PartialEq)]
pub enum SynExpr<T> {
    Atom(T),
    Deref(Box<SynExpr<T>>),
}

impl<T> From<T> for SynExpr<T> {
    fn from(atom: T) -> Self {
        SynExpr::Atom(atom)
    }
}

impl<T> SynExpr<T> {
    pub fn deref(self) -> Self {
        SynExpr::Deref(Box::new(self))
    }
}
