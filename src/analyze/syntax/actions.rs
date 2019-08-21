use super::Token;

use crate::diag::Diagnostics;
use crate::expr::BinOp;

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
pub(in crate::analyze) trait TokenStreamActions<I, L, S: Clone>: Sized {
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
// InstrLineActions::will_parse_label is called by the parser in the following state:
//
//     instr-line → . label? instr?, <Label>
//
// InstrActions as a supertrait handles the states where the label is missing (either parsing an
// instruction or terminating the empty line) whereas InstrLineActions::InstrActions handles the two
// possible states after a label has been successfully parsed. Note that by using two distinct types
// bound by InstrActions we can prevent the parser from calling InstrLineActions::will_parse_label
// more than once on the same line.
pub(in crate::analyze) trait InstrLineActions<I, L, S: Clone>:
    InstrActions<I, L, S>
{
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
pub(in crate::analyze) trait InstrActions<I, L, S: Clone>:
    LineFinalizer<S>
{
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

pub(in crate::analyze) trait LineFinalizer<S: Clone>:
    Diagnostics<S> + Sized
{
    type Next;

    fn did_parse_line(self, span: S) -> Self::Next;
}

pub(in crate::analyze) trait InstrFinalizer<S: Clone>:
    Diagnostics<S> + Sized
{
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
// BuiltinInstrActions represents any position in this rule after the initial <Ident>.
pub(in crate::analyze) trait BuiltinInstrActions<I, L, S: Clone>:
    InstrFinalizer<S>
{
    type ArgActions: ArgActions<I, L, S> + ArgFinalizer<Next = Self>;

    fn will_parse_arg(self) -> Self::ArgActions;
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
pub(in crate::analyze) trait ArgActions<I, L, S: Clone>: Diagnostics<S> {
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

pub(in crate::analyze) trait LabelActions<I, S: Clone>: Diagnostics<S> {
    type Next;

    fn act_on_param(&mut self, param: I, span: S);
    fn did_parse_label(self) -> Self::Next;
}

pub(in crate::analyze) trait MacroInstrActions<S: Clone>:
    InstrFinalizer<S>
{
    type Token;
    type MacroArgActions: MacroArgActions<S, Token = Self::Token, Next = Self>;

    fn will_parse_macro_arg(self) -> Self::MacroArgActions;
}

pub(in crate::analyze) trait MacroArgActions<S: Clone>: Diagnostics<S> {
    type Token;
    type Next;

    fn act_on_token(&mut self, token: (Self::Token, S));
    fn did_parse_macro_arg(self) -> Self::Next;
}

pub(in crate::analyze) trait TokenLineActions<I, L, S: Clone>:
    LineFinalizer<S>
{
    type ContextFinalizer: LineFinalizer<S, Next = Self::Next>;

    fn act_on_token(&mut self, token: Token<I, L>, span: S);
    fn act_on_ident(self, ident: I, span: S) -> TokenLineRule<Self, Self::ContextFinalizer>;
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
            _ => panic!("expected token sequence"),
        }
    }
}

#[cfg(test)]
pub mod mock {
    use super::*;

    use crate::diag::span::{MergeSpans, StripSpan};
    use crate::diag::{CompactDiag, DiagnosticsEvent, EmitDiag, Merge};

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum IdentKind {
        BuiltinInstr,
        Endm,
        MacroKeyword,
        MacroName,
        Other,
    }

    pub(in crate::analyze) type TokenStreamActionCollector<P, I, L, S> =
        ActionCollector<CollectedTokenStreamData<P, I, L, S>, I>;

    pub(in crate::analyze) type CollectedTokenStreamData<P, I, L, S> =
        CollectedData<TokenStreamAction<I, L, S>, LineRule<(), ()>, P>;

    impl<P, I, L, S> TokenStreamActionCollector<P, I, L, S> {
        pub fn new(parent: P, annotate: fn(&I) -> IdentKind) -> Self {
            ActionCollector {
                data: CollectedData {
                    actions: Vec::new(),
                    state: LineRule::InstrLine(()),
                    parent,
                },
                annotate,
            }
        }

        pub fn into_actions(self) -> Vec<TokenStreamAction<I, L, S>> {
            self.data.actions
        }
    }

    impl<P, I, L, S: Clone + Merge> TokenStreamActions<I, L, S>
        for TokenStreamActionCollector<P, I, L, S>
    {
        type InstrLineActions = InstrLineActionCollector<P, I, L, S>;
        type TokenLineActions = TokenLineActionCollector<P, I, L, S>;
        type TokenLineFinalizer = TokenLineActionCollector<P, I, L, S>;

        fn will_parse_line(self) -> LineRule<Self::InstrLineActions, Self::TokenLineActions> {
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

    type InstrLineActionCollector<P, I, L, S> =
        ActionCollector<InstrLineActionCollectorData<P, I, L, S>, I>;

    type InstrLineActionCollectorData<P, I, L, S> =
        CollectedData<InstrLineAction<I, L, S>, (), CollectedTokenStreamData<P, I, L, S>>;

    impl<I, L, S> From<DiagnosticsEvent<S>> for InstrLineAction<I, L, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => InstrLineAction::EmitDiag(diag),
            }
        }
    }

    impl<P, I, L, S: Clone + Merge> InstrLineActions<I, L, S> for InstrLineActionCollector<P, I, L, S> {
        type InstrActions = InstrActionCollector<P, I, L, S>;
        type LabelActions = LabelActionCollector<P, I, L, S>;

        fn will_parse_label(self, label: (I, S)) -> Self::LabelActions {
            self.push_layer(label)
        }
    }

    impl<P, I, L, S: Clone + Merge> InstrActions<I, L, S> for InstrLineActionCollector<P, I, L, S> {
        type BuiltinInstrActions = BuiltinInstrActionCollector<P, I, L, S>;
        type MacroInstrActions = MacroInstrActionCollector<P, I, L, S>;
        type ErrorActions = ErrorActionCollector<P, I, L, S>;
        type LineFinalizer = InstrActionCollector<P, I, L, S>;

        fn will_parse_instr(
            self,
            ident: I,
            span: S,
        ) -> InstrRule<Self::BuiltinInstrActions, Self::MacroInstrActions, Self::ErrorActions>
        {
            self.push_layer(()).will_parse_instr(ident, span)
        }
    }

    macro_rules! pop_layer {
        ($collector:expr) => {
            ActionCollector {
                data: $collector.data.parent,
                annotate: $collector.annotate,
            }
        };
    }

    impl<P, I, L, S: Clone + Merge> LineFinalizer<S> for InstrLineActionCollector<P, I, L, S> {
        type Next = TokenStreamActionCollector<P, I, L, S>;

        fn did_parse_line(mut self, span: S) -> Self::Next {
            self.data.parent.actions.push(TokenStreamAction::InstrLine(
                self.data.actions.split_off(0),
                span,
            ));
            pop_layer!(self)
        }
    }

    type LabelActionCollector<P, I, L, S> =
        ActionCollector<CollectedLabelActionData<P, I, L, S>, I>;

    type CollectedLabelActionData<P, I, L, S> =
        CollectedData<ParamsAction<I, S>, (I, S), InstrLineActionCollectorData<P, I, L, S>>;

    impl<I, S> From<DiagnosticsEvent<S>> for ParamsAction<I, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => ParamsAction::EmitDiag(diag),
            }
        }
    }

    impl<P, I, L, S: Clone + Merge> LabelActions<I, S> for LabelActionCollector<P, I, L, S> {
        type Next = InstrActionCollector<P, I, L, S>;

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

    type InstrActionCollector<P, I, L, S> =
        ActionCollector<CollectedInstrActionData<P, I, L, S>, I>;

    type CollectedInstrActionData<P, I, L, S> =
        CollectedData<InstrAction<I, L, S>, (), InstrLineActionCollectorData<P, I, L, S>>;

    impl<I, L, S> From<DiagnosticsEvent<S>> for InstrAction<I, L, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => InstrAction::EmitDiag(diag),
            }
        }
    }

    impl<P, I, L, S: Clone + Merge> InstrActions<I, L, S> for InstrActionCollector<P, I, L, S> {
        type BuiltinInstrActions = BuiltinInstrActionCollector<P, I, L, S>;
        type MacroInstrActions = MacroInstrActionCollector<P, I, L, S>;
        type ErrorActions = ErrorActionCollector<P, I, L, S>;
        type LineFinalizer = Self;

        fn will_parse_instr(
            self,
            ident: I,
            span: S,
        ) -> InstrRule<Self::BuiltinInstrActions, Self::MacroInstrActions, Self::ErrorActions>
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

    impl<P, I, L, S: Clone + Merge> LineFinalizer<S> for InstrActionCollector<P, I, L, S> {
        type Next = TokenStreamActionCollector<P, I, L, S>;

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

    type ErrorActionCollector<P, I, L, S> = ActionCollector<CollectedErrorData<P, I, L, S>, I>;

    type CollectedErrorData<P, I, L, S> =
        CollectedData<ErrorAction<S>, (), CollectedInstrActionData<P, I, L, S>>;

    impl<S> From<DiagnosticsEvent<S>> for ErrorAction<S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => ErrorAction::EmitDiag(diag),
            }
        }
    }

    impl<P, I, L, S: Clone + Merge> InstrFinalizer<S> for ErrorActionCollector<P, I, L, S> {
        type Next = InstrActionCollector<P, I, L, S>;

        fn did_parse_instr(mut self) -> Self::Next {
            self.data
                .parent
                .actions
                .push(InstrAction::Error(self.data.actions));
            pop_layer!(self)
        }
    }

    type BuiltinInstrActionCollector<P, I, L, S> =
        ActionCollector<CollectedBuiltinInstrActionData<P, I, L, S>, I>;

    type CollectedBuiltinInstrActionData<P, I, L, S> =
        CollectedData<BuiltinInstrAction<I, L, S>, (I, S), CollectedInstrActionData<P, I, L, S>>;

    impl<I, L, S> From<DiagnosticsEvent<S>> for BuiltinInstrAction<I, L, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => BuiltinInstrAction::EmitDiag(diag),
            }
        }
    }

    impl<P, I, L, S: Clone + Merge> BuiltinInstrActions<I, L, S>
        for BuiltinInstrActionCollector<P, I, L, S>
    {
        type ArgActions = ExprActionCollector<CollectedBuiltinInstrActionData<P, I, L, S>, I, L, S>;

        fn will_parse_arg(self) -> Self::ArgActions {
            self.push_layer(())
        }
    }

    impl<P, I, L, S: Clone + Merge> InstrFinalizer<S> for BuiltinInstrActionCollector<P, I, L, S> {
        type Next = InstrActionCollector<P, I, L, S>;

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

    pub(in crate::analyze) type ExprActionCollector<P, I, L, S> =
        ActionCollector<CollectedExprData<P, I, L, S>, I>;

    type CollectedExprData<P, I, L, S> = CollectedData<ExprAction<I, L, S>, (), P>;

    impl<I, L, S> ExprActionCollector<(), I, L, S> {
        pub fn new(annotate: fn(&I) -> IdentKind) -> Self {
            Self {
                data: CollectedData {
                    actions: Vec::new(),
                    state: (),
                    parent: (),
                },
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

    impl<P, I, L, S> ArgFinalizer
        for ExprActionCollector<CollectedBuiltinInstrActionData<P, I, L, S>, I, L, S>
    {
        type Next = BuiltinInstrActionCollector<P, I, L, S>;

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

    impl<I, L, S> ArgFinalizer for ExprActionCollector<(), I, L, S> {
        type Next = Vec<ExprAction<I, L, S>>;

        fn did_parse_arg(self) -> Self::Next {
            self.data.actions
        }
    }

    impl<P, I, L, S: Clone> ArgActions<I, L, S> for ExprActionCollector<P, I, L, S>
    where
        Self: Diagnostics<S>,
    {
        fn act_on_atom(&mut self, atom: ExprAtom<I, L>, span: S) {
            self.data.actions.push(ExprAction::PushAtom(atom, span))
        }

        fn act_on_operator(&mut self, operator: Operator, span: S) {
            self.data
                .actions
                .push(ExprAction::ApplyOperator(operator, span))
        }
    }

    type TokenLineActionCollector<P, I, L, S> =
        ActionCollector<CollectedTokenLineActionData<P, I, L, S>, I>;

    type CollectedTokenLineActionData<P, I, L, S> =
        CollectedData<TokenLineAction<I, L, S>, (), CollectedTokenStreamData<P, I, L, S>>;

    impl<I, L, S> From<DiagnosticsEvent<S>> for TokenLineAction<I, L, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => TokenLineAction::EmitDiag(diag),
            }
        }
    }

    impl<P, I, L, S: Clone + Merge> TokenLineActions<I, L, S> for TokenLineActionCollector<P, I, L, S> {
        type ContextFinalizer = Self;

        fn act_on_token(&mut self, token: Token<I, L>, span: S) {
            self.data
                .actions
                .push(TokenLineAction::Token((token, span)))
        }

        fn act_on_ident(
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

    impl<P, I, L, S: Clone + Merge> LineFinalizer<S> for TokenLineActionCollector<P, I, L, S> {
        type Next = TokenStreamActionCollector<P, I, L, S>;

        fn did_parse_line(mut self, span: S) -> Self::Next {
            self.data
                .parent
                .actions
                .push(TokenStreamAction::TokenLine(self.data.actions, span));
            pop_layer!(self)
        }
    }

    type MacroInstrActionCollector<P, I, L, S> =
        ActionCollector<CollectedMacroInstrData<P, I, L, S>, I>;

    type CollectedMacroInstrData<P, I, L, S> =
        CollectedData<MacroInstrAction<I, L, S>, (I, S), CollectedInstrActionData<P, I, L, S>>;

    impl<I, L, S> From<DiagnosticsEvent<S>> for MacroInstrAction<I, L, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => MacroInstrAction::EmitDiag(diag),
            }
        }
    }

    impl<P, I, L, S: Clone + Merge> MacroInstrActions<S> for MacroInstrActionCollector<P, I, L, S> {
        type Token = Token<I, L>;
        type MacroArgActions = MacroArgActionCollector<P, I, L, S>;

        fn will_parse_macro_arg(self) -> Self::MacroArgActions {
            self.push_layer(())
        }
    }

    impl<P, I, L, S: Clone + Merge> InstrFinalizer<S> for MacroInstrActionCollector<P, I, L, S> {
        type Next = InstrActionCollector<P, I, L, S>;

        fn did_parse_instr(mut self) -> Self::Next {
            self.data.parent.actions.push(InstrAction::MacroInstr {
                name: self.data.state,
                actions: self.data.actions,
            });
            pop_layer!(self)
        }
    }

    type MacroArgActionCollector<P, I, L, S> =
        ActionCollector<CollectedMacroArgData<P, I, L, S>, I>;

    type CollectedMacroArgData<P, I, L, S> =
        CollectedData<TokenSeqAction<I, L, S>, (), CollectedMacroInstrData<P, I, L, S>>;

    impl<I, L, S> From<DiagnosticsEvent<S>> for TokenSeqAction<I, L, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            match event {
                DiagnosticsEvent::EmitDiag(diag) => TokenSeqAction::EmitDiag(diag),
            }
        }
    }

    impl<P, I, L, S: Clone + Merge> MacroArgActions<S> for MacroArgActionCollector<P, I, L, S> {
        type Token = Token<I, L>;
        type Next = MacroInstrActionCollector<P, I, L, S>;

        fn act_on_token(&mut self, token: (Self::Token, S)) {
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

    pub(in crate::analyze) struct ActionCollector<D, I> {
        data: D,
        annotate: fn(&I) -> IdentKind,
    }

    pub(in crate::analyze) struct CollectedData<A, T, P> {
        actions: Vec<A>,
        state: T,
        parent: P,
    }

    impl<A1, T1, P, I> ActionCollector<CollectedData<A1, T1, P>, I> {
        fn push_layer<A2, T2>(
            self,
            state: T2,
        ) -> ActionCollector<NestedCollectedData<A1, A2, T1, T2, P>, I> {
            ActionCollector {
                data: CollectedData {
                    actions: Vec::new(),
                    state,
                    parent: self.data,
                },
                annotate: self.annotate,
            }
        }
    }

    type NestedCollectedData<A1, A2, T1, T2, P> = CollectedData<A2, T2, CollectedData<A1, T1, P>>;

    impl<D, I, S: Clone + Merge> MergeSpans<S> for ActionCollector<D, I> {
        fn merge_spans(&mut self, left: &S, right: &S) -> S {
            S::merge(left.clone(), right.clone())
        }
    }

    impl<D, I, S: Clone> StripSpan<S> for ActionCollector<D, I> {
        type Stripped = S;

        fn strip_span(&mut self, span: &S) -> Self::Stripped {
            span.clone()
        }
    }

    impl<A, T, P, I, S> EmitDiag<S, S> for ActionCollector<CollectedData<A, T, P>, I>
    where
        A: From<DiagnosticsEvent<S>>,
        S: Clone,
    {
        fn emit_diag(&mut self, diag: impl Into<CompactDiag<S, S>>) {
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
