use super::Token;

use crate::diag::CompactDiag;
use crate::expr::BinOp;

pub(in crate::analyze) trait ParsingContext: Sized {
    type Ident;
    type Literal;
    type Error;
    type Span;
    type Stripped;

    fn next_token(
        &mut self,
    ) -> Option<LexerOutput<Self::Ident, Self::Literal, Self::Error, Self::Span>>;

    fn merge_spans(&mut self, left: &Self::Span, right: &Self::Span) -> Self::Span;
    fn strip_span(&mut self, span: &Self::Span) -> Self::Stripped;
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<Self::Span, Self::Stripped>>);
}

pub(in crate::analyze) type LexerOutput<I, L, E, S> = (Result<Token<I, L>, E>, S);

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
// TokenStreamContext::will_parse_line.
pub(in crate::analyze) trait TokenStreamContext: ParsingContext {
    type InstrLineContext: InstrLineContext<
        Ident = Self::Ident,
        Literal = Self::Literal,
        Error = Self::Error,
        Span = Self::Span,
        Stripped = Self::Stripped,
        Next = Self,
    >;
    type TokenLineContext: TokenLineContext<
        Ident = Self::Ident,
        Literal = Self::Literal,
        Error = Self::Error,
        Span = Self::Span,
        Stripped = Self::Stripped,
        ContextFinalizer = Self::TokenLineFinalizer,
        Next = Self,
    >;
    type TokenLineFinalizer: LineFinalizer<
        Ident = Self::Ident,
        Literal = Self::Literal,
        Error = Self::Error,
        Span = Self::Span,
        Stripped = Self::Stripped,
        Next = Self,
    >;

    fn will_parse_line(self) -> LineRule<Self::InstrLineContext, Self::TokenLineContext>;
    fn act_on_eos(self, span: Self::Span) -> Self;
}

#[derive(Debug, PartialEq)]
pub(in crate::analyze) enum LineRule<I, T> {
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
// InstrLineContext::will_parse_label is called by the parser in the following state:
//
//     instr-line → . label? instr?, <Label>
//
// InstrContext as a supertrait handles the states where the label is missing (either parsing an
// instruction or terminating the empty line) whereas InstrLineContext::InstrContext handles the two
// possible states after a label has been successfully parsed. Note that by using two distinct types
// bound by InstrContext we can prevent the parser from calling InstrLineContext::will_parse_label
// more than once on the same line.
pub(in crate::analyze) trait InstrLineContext: InstrContext {
    type LabelContext: LabelContext<
        Ident = Self::Ident,
        Literal = Self::Literal,
        Error = Self::Error,
        Span = Self::Span,
        Stripped = Self::Stripped,
        Next = Self::InstrContext,
    >;
    type InstrContext: InstrContext<
        Ident = Self::Ident,
        Literal = Self::Literal,
        Error = Self::Error,
        Span = Self::Span,
        Stripped = Self::Stripped,
        Next = Self::Next,
    >;

    fn will_parse_label(self, label: (Self::Ident, Self::Span)) -> Self::LabelContext;
}

// An instruction can be either a builtin instruction (i.e. a CPU instruction or an assembler
// directive) or a macro instruction previously defined by the program. These two options correspond
// to two production rules:
//
//     1. instr → builtin-instr
//     2. instr → macro-instr
//
// The ambiguity between these rules gets resolved by InstrContext::will_parse_instr, which performs
// a name lookup to determine whether the identifier is a builtin instruction or a previously
// defined macro. If neither of these cases applies, a third production rule is used:
//
//     3. instr → <Ident> token-seq
//
// The parser uses this rule to recover from an invalid instruction name by throwing away all the
// remaining tokens in the line.
pub(in crate::analyze) trait InstrContext: LineFinalizer {
    type BuiltinInstrContext: BuiltinInstrContext<
        Ident = Self::Ident,
        Literal = Self::Literal,
        Error = Self::Error,
        Span = Self::Span,
        Stripped = Self::Stripped,
        Next = Self::LineFinalizer,
    >;
    type MacroInstrContext: MacroInstrContext<
        Ident = Self::Ident,
        Literal = Self::Literal,
        Error = Self::Error,
        Span = Self::Span,
        Stripped = Self::Stripped,
        Next = Self::LineFinalizer,
    >;
    type ErrorContext: InstrFinalizer<
        Ident = Self::Ident,
        Literal = Self::Literal,
        Error = Self::Error,
        Span = Self::Span,
        Stripped = Self::Stripped,
        Next = Self::LineFinalizer,
    >;
    type LineFinalizer: LineFinalizer<
        Ident = Self::Ident,
        Literal = Self::Literal,
        Error = Self::Error,
        Span = Self::Span,
        Stripped = Self::Stripped,
        Next = Self::Next,
    >;

    fn will_parse_instr(
        self,
        ident: Self::Ident,
        span: Self::Span,
    ) -> InstrRule<Self::BuiltinInstrContext, Self::MacroInstrContext, Self::ErrorContext>;
}

pub(in crate::analyze) trait LineFinalizer: ParsingContext {
    type Next;

    fn did_parse_line(self, span: Self::Span) -> Self::Next;
}

pub(in crate::analyze) trait InstrFinalizer: ParsingContext {
    type Next;

    fn did_parse_instr(self) -> Self::Next;
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum InstrRule<C, M, E> {
    BuiltinInstr(C),
    MacroInstr(M),
    Error(E),
}

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
// BuiltinInstrContext represents any position in this rule after the initial <Ident>.
pub(in crate::analyze) trait BuiltinInstrContext: InstrFinalizer {
    type ArgContext: ArgContext<
            Ident = Self::Ident,
            Literal = Self::Literal,
            Error = Self::Error,
            Span = Self::Span,
            Stripped = Self::Stripped,
        > + ArgFinalizer<Next = Self>;

    fn will_parse_arg(self) -> Self::ArgContext;
}

pub(in crate::analyze) trait ArgFinalizer {
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
pub(in crate::analyze) trait ArgContext: ParsingContext {
    fn act_on_atom(&mut self, atom: ExprAtom<Self::Ident, Self::Literal>, span: Self::Span);
    fn act_on_operator(&mut self, operator: Operator, span: Self::Span);
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

pub(in crate::analyze) trait LabelContext: ParsingContext {
    type Next;

    fn act_on_param(&mut self, param: Self::Ident, span: Self::Span);
    fn did_parse_label(self) -> Self::Next;
}

pub(in crate::analyze) trait MacroInstrContext: InstrFinalizer {
    type MacroArgContext: MacroArgContext<
        Ident = Self::Ident,
        Literal = Self::Literal,
        Error = Self::Error,
        Span = Self::Span,
        Stripped = Self::Stripped,
        Next = Self,
    >;

    fn will_parse_macro_arg(self) -> Self::MacroArgContext;
}

pub(in crate::analyze) trait MacroArgContext: ParsingContext {
    type Next;

    fn act_on_token(&mut self, token: (Token<Self::Ident, Self::Literal>, Self::Span));
    fn did_parse_macro_arg(self) -> Self::Next;
}

pub(in crate::analyze) trait TokenLineContext: LineFinalizer {
    type ContextFinalizer: LineFinalizer<
        Ident = Self::Ident,
        Literal = Self::Literal,
        Error = Self::Error,
        Span = Self::Span,
        Stripped = Self::Stripped,
        Next = Self::Next,
    >;

    fn act_on_token(&mut self, token: Token<Self::Ident, Self::Literal>, span: Self::Span);
    fn act_on_mnemonic(
        self,
        ident: Self::Ident,
        span: Self::Span,
    ) -> TokenLineRule<Self, Self::ContextFinalizer>;
}

pub(in crate::analyze) enum TokenLineRule<T, E> {
    TokenSeq(T),
    LineEnd(E),
}

#[cfg(test)]
impl<T, E> TokenLineRule<T, E> {
    pub fn into_line_end(self) -> E {
        match self {
            TokenLineRule::LineEnd(context) => context,
            _ => panic!("expected line end"),
        }
    }
}

#[cfg(test)]
pub mod mock {
    use super::*;

    use crate::diag::span::{MergeSpans, StripSpan};
    use crate::diag::{CompactDiag, DiagnosticsEvent, Merge};

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum IdentKind {
        BuiltinInstr,
        Endm,
        MacroKeyword,
        MacroName,
        Other,
    }

    pub(in crate::analyze) type TokenStreamActionCollector<'a, P, I, L, E, S> =
        ActionCollector<'a, CollectedTokenStreamData<P, I, L, S>, I, L, E, S>;

    pub(in crate::analyze) type CollectedTokenStreamData<P, I, L, S> =
        CollectedData<TokenStreamAction<I, L, S>, LineRule<(), ()>, P>;

    impl<'a, P, I, L, E, S> TokenStreamActionCollector<'a, P, I, L, E, S> {
        pub fn new(
            parent: P,
            tokens: &'a mut dyn Iterator<Item = LexerOutput<I, L, E, S>>,
            annotate: fn(&I) -> IdentKind,
        ) -> Self {
            ActionCollector {
                data: CollectedData {
                    actions: Vec::new(),
                    state: LineRule::InstrLine(()),
                    parent,
                },
                tokens,
                annotate,
            }
        }

        pub fn into_actions(self) -> Vec<TokenStreamAction<I, L, S>> {
            self.data.actions
        }
    }

    impl<I, L, S> From<DiagnosticsEvent<S>> for TokenStreamAction<I, L, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => TokenStreamAction::EmitDiag(diag),
            }
        }
    }

    impl<'a, P, I, L, E, S: Clone + Merge> TokenStreamContext
        for TokenStreamActionCollector<'a, P, I, L, E, S>
    {
        type InstrLineContext = InstrLineActionCollector<'a, P, I, L, E, S>;
        type TokenLineContext = TokenLineActionCollector<'a, P, I, L, E, S>;
        type TokenLineFinalizer = TokenLineActionCollector<'a, P, I, L, E, S>;

        fn will_parse_line(self) -> LineRule<Self::InstrLineContext, Self::TokenLineContext> {
            match self.data.state {
                LineRule::InstrLine(()) => LineRule::InstrLine(self.push_layer(())),
                LineRule::TokenLine(()) => LineRule::TokenLine(self.push_layer(())),
            }
        }

        fn act_on_eos(mut self, span: S) -> Self {
            self.data.actions.push(TokenStreamAction::Eos(span));
            self
        }
    }

    type InstrLineActionCollector<'a, P, I, L, E, S> =
        ActionCollector<'a, InstrLineActionCollectorData<P, I, L, S>, I, L, E, S>;

    type InstrLineActionCollectorData<P, I, L, S> =
        CollectedData<InstrLineAction<I, L, S>, (), CollectedTokenStreamData<P, I, L, S>>;

    impl<I, L, S> From<DiagnosticsEvent<S>> for InstrLineAction<I, L, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => InstrLineAction::EmitDiag(diag),
            }
        }
    }

    impl<'a, P, I, L, E, S: Clone + Merge> InstrLineContext
        for InstrLineActionCollector<'a, P, I, L, E, S>
    {
        type InstrContext = InstrActionCollector<'a, P, I, L, E, S>;
        type LabelContext = LabelActionCollector<'a, P, I, L, E, S>;

        fn will_parse_label(self, label: (I, S)) -> Self::LabelContext {
            self.push_layer(label)
        }
    }

    impl<'a, P, I, L, E, S: Clone + Merge> InstrContext
        for InstrLineActionCollector<'a, P, I, L, E, S>
    {
        type BuiltinInstrContext = BuiltinInstrActionCollector<'a, P, I, L, E, S>;
        type MacroInstrContext = MacroInstrActionCollector<'a, P, I, L, E, S>;
        type ErrorContext = ErrorActionCollector<'a, P, I, L, E, S>;
        type LineFinalizer = InstrActionCollector<'a, P, I, L, E, S>;

        fn will_parse_instr(
            self,
            ident: I,
            span: S,
        ) -> InstrRule<Self::BuiltinInstrContext, Self::MacroInstrContext, Self::ErrorContext>
        {
            self.push_layer(()).will_parse_instr(ident, span)
        }
    }

    macro_rules! pop_layer {
        ($collector:expr) => {
            ActionCollector {
                data: $collector.data.parent,
                tokens: $collector.tokens,
                annotate: $collector.annotate,
            }
        };
    }

    impl<'a, P, I, L, E, S: Clone + Merge> LineFinalizer
        for InstrLineActionCollector<'a, P, I, L, E, S>
    {
        type Next = TokenStreamActionCollector<'a, P, I, L, E, S>;

        fn did_parse_line(mut self, span: S) -> Self::Next {
            self.data.parent.actions.push(TokenStreamAction::InstrLine(
                self.data.actions.split_off(0),
                span,
            ));
            pop_layer!(self)
        }
    }

    type LabelActionCollector<'a, P, I, L, E, S> =
        ActionCollector<'a, CollectedLabelActionData<P, I, L, S>, I, L, E, S>;

    type CollectedLabelActionData<P, I, L, S> =
        CollectedData<ParamsAction<I, S>, (I, S), InstrLineActionCollectorData<P, I, L, S>>;

    impl<I, S> From<DiagnosticsEvent<S>> for ParamsAction<I, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => ParamsAction::EmitDiag(diag),
            }
        }
    }

    impl<'a, P, I, L, E, S: Clone + Merge> LabelContext for LabelActionCollector<'a, P, I, L, E, S> {
        type Next = InstrActionCollector<'a, P, I, L, E, S>;

        fn act_on_param(&mut self, param: I, span: S) {
            self.data
                .actions
                .push(ParamsAction::AddParameter(param, span))
        }

        fn did_parse_label(mut self) -> Self::Next {
            self.data
                .parent
                .actions
                .push(InstrLineAction::Label((self.data.state, self.data.actions)));
            pop_layer!(self).push_layer(())
        }
    }

    type InstrActionCollector<'a, P, I, L, E, S> =
        ActionCollector<'a, CollectedInstrActionData<P, I, L, S>, I, L, E, S>;

    type CollectedInstrActionData<P, I, L, S> =
        CollectedData<InstrAction<I, L, S>, (), InstrLineActionCollectorData<P, I, L, S>>;

    impl<I, L, S> From<DiagnosticsEvent<S>> for InstrAction<I, L, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => InstrAction::EmitDiag(diag),
            }
        }
    }

    impl<'a, P, I, L, E, S: Clone + Merge> InstrContext for InstrActionCollector<'a, P, I, L, E, S> {
        type BuiltinInstrContext = BuiltinInstrActionCollector<'a, P, I, L, E, S>;
        type MacroInstrContext = MacroInstrActionCollector<'a, P, I, L, E, S>;
        type ErrorContext = ErrorActionCollector<'a, P, I, L, E, S>;
        type LineFinalizer = Self;

        fn will_parse_instr(
            self,
            ident: I,
            span: S,
        ) -> InstrRule<Self::BuiltinInstrContext, Self::MacroInstrContext, Self::ErrorContext>
        {
            match (self.annotate)(&ident) {
                IdentKind::BuiltinInstr | IdentKind::MacroKeyword | IdentKind::Endm => {
                    InstrRule::BuiltinInstr(self.push_layer((ident, span)))
                }
                IdentKind::MacroName => InstrRule::MacroInstr(self.push_layer((ident, span))),
                IdentKind::Other => InstrRule::Error(self.push_layer(())),
            }
        }
    }

    impl<'a, P, I, L, E, S: Clone + Merge> LineFinalizer for InstrActionCollector<'a, P, I, L, E, S> {
        type Next = TokenStreamActionCollector<'a, P, I, L, E, S>;

        fn did_parse_line(mut self, span: S) -> Self::Next {
            if !self.data.actions.is_empty() {
                self.data
                    .parent
                    .actions
                    .push(InstrLineAction::Instr(self.data.actions));
            }
            pop_layer!(self).did_parse_line(span)
        }
    }

    type ErrorActionCollector<'a, P, I, L, E, S> =
        ActionCollector<'a, CollectedErrorData<P, I, L, S>, I, L, E, S>;

    type CollectedErrorData<P, I, L, S> =
        CollectedData<ErrorAction<S>, (), CollectedInstrActionData<P, I, L, S>>;

    impl<S> From<DiagnosticsEvent<S>> for ErrorAction<S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => ErrorAction::EmitDiag(diag),
            }
        }
    }

    impl<'a, P, I, L, E, S: Clone + Merge> InstrFinalizer for ErrorActionCollector<'a, P, I, L, E, S> {
        type Next = InstrActionCollector<'a, P, I, L, E, S>;

        fn did_parse_instr(mut self) -> Self::Next {
            self.data
                .parent
                .actions
                .push(InstrAction::Error(self.data.actions));
            pop_layer!(self)
        }
    }

    type BuiltinInstrActionCollector<'a, P, I, L, E, S> =
        ActionCollector<'a, CollectedBuiltinInstrActionData<P, I, L, S>, I, L, E, S>;

    type CollectedBuiltinInstrActionData<P, I, L, S> =
        CollectedData<BuiltinInstrAction<I, L, S>, (I, S), CollectedInstrActionData<P, I, L, S>>;

    impl<I, L, S> From<DiagnosticsEvent<S>> for BuiltinInstrAction<I, L, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => BuiltinInstrAction::EmitDiag(diag),
            }
        }
    }

    impl<'a, P, I, L, E, S: Clone + Merge> BuiltinInstrContext
        for BuiltinInstrActionCollector<'a, P, I, L, E, S>
    {
        type ArgContext =
            ExprActionCollector<'a, CollectedBuiltinInstrActionData<P, I, L, S>, I, L, E, S>;

        fn will_parse_arg(self) -> Self::ArgContext {
            self.push_layer(())
        }
    }

    impl<'a, P, I, L, E, S: Clone + Merge> InstrFinalizer
        for BuiltinInstrActionCollector<'a, P, I, L, E, S>
    {
        type Next = InstrActionCollector<'a, P, I, L, E, S>;

        fn did_parse_instr(mut self) -> Self::Next {
            if (self.annotate)(&self.data.state.0) == IdentKind::MacroKeyword {
                self.data.parent.parent.parent.state = LineRule::TokenLine(())
            }
            self.data.parent.actions.push(InstrAction::BuiltinInstr {
                builtin_instr: self.data.state,
                actions: self.data.actions,
            });
            pop_layer!(self)
        }
    }

    pub(in crate::analyze) type ExprActionCollector<'a, P, I, L, E, S> =
        ActionCollector<'a, CollectedExprData<P, I, L, S>, I, L, E, S>;

    type CollectedExprData<P, I, L, S> = CollectedData<ExprAction<I, L, S>, (), P>;

    impl<'a, I, L, E, S> ExprActionCollector<'a, (), I, L, E, S> {
        pub fn new(
            tokens: &'a mut dyn Iterator<Item = LexerOutput<I, L, E, S>>,
            annotate: fn(&I) -> IdentKind,
        ) -> Self {
            Self {
                data: CollectedData {
                    actions: Vec::new(),
                    state: (),
                    parent: (),
                },
                tokens,
                annotate,
            }
        }
    }

    impl<I, L, S> From<DiagnosticsEvent<S>> for ExprAction<I, L, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => ExprAction::EmitDiag(diag),
            }
        }
    }

    impl<'a, P, I, L, E, S> ArgFinalizer
        for ExprActionCollector<'a, CollectedBuiltinInstrActionData<P, I, L, S>, I, L, E, S>
    {
        type Next = BuiltinInstrActionCollector<'a, P, I, L, E, S>;

        fn did_parse_arg(mut self) -> Self::Next {
            self.data
                .parent
                .actions
                .push(BuiltinInstrAction::AddArgument {
                    actions: self.data.actions,
                });
            pop_layer!(self)
        }
    }

    impl<'a, I, L, E, S> ArgFinalizer for ExprActionCollector<'a, (), I, L, E, S> {
        type Next = Vec<ExprAction<I, L, S>>;

        fn did_parse_arg(self) -> Self::Next {
            self.data.actions
        }
    }

    impl<'a, P, I, L, E, S: Clone + Merge> ArgContext for ExprActionCollector<'a, P, I, L, E, S> {
        fn act_on_atom(&mut self, atom: ExprAtom<I, L>, span: S) {
            self.data.actions.push(ExprAction::PushAtom(atom, span))
        }

        fn act_on_operator(&mut self, operator: Operator, span: S) {
            self.data
                .actions
                .push(ExprAction::ApplyOperator(operator, span))
        }
    }

    type TokenLineActionCollector<'a, P, I, L, E, S> =
        ActionCollector<'a, CollectedTokenLineActionData<P, I, L, S>, I, L, E, S>;

    type CollectedTokenLineActionData<P, I, L, S> =
        CollectedData<TokenLineAction<I, L, S>, (), CollectedTokenStreamData<P, I, L, S>>;

    impl<I, L, S> From<DiagnosticsEvent<S>> for TokenLineAction<I, L, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => TokenLineAction::EmitDiag(diag),
            }
        }
    }

    impl<'a, P, I, L, E, S: Clone + Merge> TokenLineContext
        for TokenLineActionCollector<'a, P, I, L, E, S>
    {
        type ContextFinalizer = Self;

        fn act_on_token(&mut self, token: Token<I, L>, span: S) {
            self.data
                .actions
                .push(TokenLineAction::Token((token, span)))
        }

        fn act_on_mnemonic(
            mut self,
            ident: I,
            span: S,
        ) -> TokenLineRule<Self, Self::ContextFinalizer> {
            let kind = (self.annotate)(&ident);
            self.data
                .actions
                .push(TokenLineAction::Ident((ident, span)));
            match kind {
                IdentKind::Endm => {
                    self.data.parent.state = LineRule::InstrLine(());
                    TokenLineRule::LineEnd(self)
                }
                _ => TokenLineRule::TokenSeq(self),
            }
        }
    }

    impl<'a, P, I, L, E, S: Clone + Merge> LineFinalizer
        for TokenLineActionCollector<'a, P, I, L, E, S>
    {
        type Next = TokenStreamActionCollector<'a, P, I, L, E, S>;

        fn did_parse_line(mut self, span: S) -> Self::Next {
            self.data
                .parent
                .actions
                .push(TokenStreamAction::TokenLine(self.data.actions, span));
            pop_layer!(self)
        }
    }

    type MacroInstrActionCollector<'a, P, I, L, E, S> =
        ActionCollector<'a, CollectedMacroInstrData<P, I, L, S>, I, L, E, S>;

    type CollectedMacroInstrData<P, I, L, S> =
        CollectedData<MacroInstrAction<I, L, S>, (I, S), CollectedInstrActionData<P, I, L, S>>;

    impl<I, L, S> From<DiagnosticsEvent<S>> for MacroInstrAction<I, L, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => MacroInstrAction::EmitDiag(diag),
            }
        }
    }

    impl<'a, P, I, L, E, S: Clone + Merge> MacroInstrContext
        for MacroInstrActionCollector<'a, P, I, L, E, S>
    {
        type MacroArgContext = MacroArgActionCollector<'a, P, I, L, E, S>;

        fn will_parse_macro_arg(self) -> Self::MacroArgContext {
            self.push_layer(())
        }
    }

    impl<'a, P, I, L, E, S: Clone + Merge> InstrFinalizer
        for MacroInstrActionCollector<'a, P, I, L, E, S>
    {
        type Next = InstrActionCollector<'a, P, I, L, E, S>;

        fn did_parse_instr(mut self) -> Self::Next {
            self.data.parent.actions.push(InstrAction::MacroInstr {
                name: self.data.state,
                actions: self.data.actions,
            });
            pop_layer!(self)
        }
    }

    type MacroArgActionCollector<'a, P, I, L, E, S> =
        ActionCollector<'a, CollectedMacroArgData<P, I, L, S>, I, L, E, S>;

    type CollectedMacroArgData<P, I, L, S> =
        CollectedData<TokenSeqAction<I, L, S>, (), CollectedMacroInstrData<P, I, L, S>>;

    impl<I, L, S> From<DiagnosticsEvent<S>> for TokenSeqAction<I, L, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => TokenSeqAction::EmitDiag(diag),
            }
        }
    }

    impl<'a, P, I, L, E, S: Clone + Merge> MacroArgContext
        for MacroArgActionCollector<'a, P, I, L, E, S>
    {
        type Next = MacroInstrActionCollector<'a, P, I, L, E, S>;

        fn act_on_token(&mut self, token: (Token<I, L>, S)) {
            self.data.actions.push(TokenSeqAction::PushToken(token))
        }

        fn did_parse_macro_arg(mut self) -> Self::Next {
            self.data
                .parent
                .actions
                .push(MacroInstrAction::MacroArg(self.data.actions));
            pop_layer!(self)
        }
    }

    pub(in crate::analyze) struct ActionCollector<'a, D, I, L, E, S> {
        pub data: D,
        tokens: &'a mut dyn Iterator<Item = LexerOutput<I, L, E, S>>,
        annotate: fn(&I) -> IdentKind,
    }

    pub(in crate::analyze) struct CollectedData<A, T, P> {
        actions: Vec<A>,
        state: T,
        pub parent: P,
    }

    impl<'a, A1, T1, P, I, L, E, S> ActionCollector<'a, CollectedData<A1, T1, P>, I, L, E, S> {
        fn push_layer<A2, T2>(
            self,
            state: T2,
        ) -> ActionCollector<'a, NestedCollectedData<A1, A2, T1, T2, P>, I, L, E, S> {
            ActionCollector {
                data: CollectedData {
                    actions: Vec::new(),
                    state,
                    parent: self.data,
                },
                tokens: self.tokens,
                annotate: self.annotate,
            }
        }
    }

    type NestedCollectedData<A1, A2, T1, T2, P> = CollectedData<A2, T2, CollectedData<A1, T1, P>>;

    impl<'a, D, I, L, E, S: Clone + Merge> MergeSpans<S> for ActionCollector<'a, D, I, L, E, S> {
        fn merge_spans(&mut self, left: &S, right: &S) -> S {
            S::merge(left.clone(), right.clone())
        }
    }

    impl<'a, D, I, L, E, S: Clone> StripSpan<S> for ActionCollector<'a, D, I, L, E, S> {
        type Stripped = S;

        fn strip_span(&mut self, span: &S) -> Self::Stripped {
            span.clone()
        }
    }

    impl<'a, A, T, P, I, L, E, S> ParsingContext
        for ActionCollector<'a, CollectedData<A, T, P>, I, L, E, S>
    where
        A: From<DiagnosticsEvent<S>>,
        S: Clone + Merge,
    {
        type Ident = I;
        type Literal = L;
        type Error = E;
        type Span = S;
        type Stripped = S;

        fn next_token(&mut self) -> Option<LexerOutput<I, L, E, S>> {
            self.tokens.next()
        }

        fn merge_spans(&mut self, left: &S, right: &S) -> S {
            S::merge(left.clone(), right.clone())
        }

        fn strip_span(&mut self, span: &S) -> Self::Stripped {
            span.clone()
        }

        fn emit_diag(&mut self, diag: impl Into<CompactDiag<Self::Span, Self::Stripped>>) {
            self.data
                .actions
                .push(DiagnosticsEvent::EmitDiag(diag.into()).into())
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    pub(in crate::analyze) enum TokenStreamAction<I, L, S> {
        InstrLine(Vec<InstrLineAction<I, L, S>>, S),
        TokenLine(Vec<TokenLineAction<I, L, S>>, S),
        Eos(S),
        EmitDiag(CompactDiag<S>),
    }

    #[derive(Clone, Debug, PartialEq)]
    pub(in crate::analyze) enum InstrLineAction<I, L, S> {
        Label(Label<I, S>),
        Instr(Vec<InstrAction<I, L, S>>),
        EmitDiag(CompactDiag<S>),
    }

    pub(in crate::analyze) type Label<I, S> = ((I, S), Vec<ParamsAction<I, S>>);

    #[derive(Clone, Debug, PartialEq)]
    pub(in crate::analyze) enum InstrAction<I, L, S> {
        BuiltinInstr {
            builtin_instr: (I, S),
            actions: Vec<BuiltinInstrAction<I, L, S>>,
        },
        MacroInstr {
            name: (I, S),
            actions: Vec<MacroInstrAction<I, L, S>>,
        },
        Error(Vec<ErrorAction<S>>),
        EmitDiag(CompactDiag<S>),
    }

    #[derive(Clone, Debug, PartialEq)]
    pub(in crate::analyze) enum BuiltinInstrAction<I, L, S> {
        AddArgument { actions: Vec<ExprAction<I, L, S>> },
        EmitDiag(CompactDiag<S>),
    }

    #[derive(Clone, Debug, PartialEq)]
    pub(in crate::analyze) enum ExprAction<I, L, S> {
        PushAtom(ExprAtom<I, L>, S),
        ApplyOperator(Operator, S),
        EmitDiag(CompactDiag<S>),
    }

    #[derive(Clone, Debug, PartialEq)]
    pub(in crate::analyze) enum ParamsAction<I, S> {
        AddParameter(I, S),
        EmitDiag(CompactDiag<S>),
    }

    #[derive(Clone, Debug, PartialEq)]
    pub(in crate::analyze) enum TokenLineAction<I, L, S> {
        Token((Token<I, L>, S)),
        Ident((I, S)),
        EmitDiag(CompactDiag<S>),
    }

    #[derive(Clone, Debug, PartialEq)]
    pub(in crate::analyze) enum TokenSeqAction<I, L, S> {
        PushToken((Token<I, L>, S)),
        EmitDiag(CompactDiag<S>),
    }

    #[derive(Clone, Debug, PartialEq)]
    pub(in crate::analyze) enum ErrorAction<S> {
        EmitDiag(CompactDiag<S>),
    }

    #[derive(Clone, Debug, PartialEq)]
    pub(in crate::analyze) enum MacroInstrAction<I, L, S> {
        MacroArg(Vec<TokenSeqAction<I, L, S>>),
        EmitDiag(CompactDiag<S>),
    }
}
