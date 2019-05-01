use self::SimpleToken::*;

use super::{Parser, LINE_FOLLOW_SET};

use crate::diag::span::{MergeSpans, StripSpan};
use crate::diag::{CompactDiagnostic, DelegateDiagnostics, EmitDiagnostic, Message};
use crate::model::BinOp;
use crate::syntax::{ExprAtom, ExprContext, Operator, SimpleToken, Token, UnaryOperator};

type ParserResult<P, C, S> = Result<
    P,
    (
        P,
        ExpandedExprParsingError<<C as DelegateDiagnostics<S>>::Delegate, S>,
    ),
>;

type ExpandedExprParsingError<D, S> = ExprParsingError<S, <D as StripSpan<S>>::Stripped>;

enum ExprParsingError<S, R> {
    NothingParsed,
    Other(CompactDiagnostic<S, R>),
}

enum SuffixOperator {
    Binary(BinOp),
    FnCall,
}

impl<I, C, L> Token<I, C, L> {
    fn as_suffix_operator(&self) -> Option<SuffixOperator> {
        use SuffixOperator::*;
        match self {
            Token::Simple(Minus) => Some(Binary(BinOp::Minus)),
            Token::Simple(LParen) => Some(FnCall),
            Token::Simple(Pipe) => Some(Binary(BinOp::BitwiseOr)),
            Token::Simple(Plus) => Some(Binary(BinOp::Plus)),
            Token::Simple(Slash) => Some(Binary(BinOp::Division)),
            Token::Simple(Star) => Some(Binary(BinOp::Multiplication)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
enum Precedence {
    None,
    BitwiseOr,
    Addition,
    Multiplication,
    FnCall,
}

impl SuffixOperator {
    fn precedence(&self) -> Precedence {
        use SuffixOperator::*;
        match self {
            Binary(BinOp::BitwiseOr) => Precedence::BitwiseOr,
            Binary(BinOp::Plus) | Binary(BinOp::Minus) => Precedence::Addition,
            Binary(BinOp::Multiplication) | Binary(BinOp::Division) => Precedence::Multiplication,
            FnCall => Precedence::FnCall,
        }
    }
}

impl<'a, Id, L, C, E, I, Ctx, S> Parser<'a, (Result<Token<Id, L, C>, E>, S), I, Ctx>
where
    I: Iterator<Item = (Result<Token<Id, L, C>, E>, S)>,
    Ctx: ExprContext<S, Ident = Id, Literal = L>,
    S: Clone,
{
    pub(super) fn parse(self) -> Self {
        self.parse_expression()
            .unwrap_or_else(|(mut parser, error)| {
                match error {
                    ExprParsingError::NothingParsed => parser = parser.diagnose_unexpected_token(),
                    ExprParsingError::Other(diagnostic) => parser.emit_diagnostic(diagnostic),
                }
                while !parser.token_is_in(LINE_FOLLOW_SET) {
                    bump!(parser);
                }
                parser
            })
    }

    fn parse_expression(self) -> ParserResult<Self, Ctx, S> {
        self.parse_infix_expr(Precedence::None)
    }

    fn parse_parenthesized_expression(mut self, left: S) -> ParserResult<Self, Ctx, S> {
        self = match self.parse_expression() {
            Ok(parser) => parser,
            Err((parser, error)) => {
                let error = match error {
                    error @ ExprParsingError::NothingParsed => match parser.token.0 {
                        Ok(Token::Simple(Eof)) | Ok(Token::Simple(Eol)) => {
                            ExprParsingError::Other(Message::UnmatchedParenthesis.at(left).into())
                        }
                        _ => error,
                    },
                    error => error,
                };
                return Err((parser, error));
            }
        };
        match self.token {
            (Ok(Token::Simple(RParen)), right) => {
                bump!(self);
                let span = self.context.diagnostics().merge_spans(&left, &right);
                self.context
                    .apply_operator((Operator::Unary(UnaryOperator::Parentheses), span));
                Ok(self)
            }
            _ => Err((
                self,
                ExprParsingError::Other(Message::UnmatchedParenthesis.at(left).into()),
            )),
        }
    }

    fn parse_infix_expr(mut self, lowest: Precedence) -> ParserResult<Self, Ctx, S> {
        self = self.parse_primary_expr()?;
        while let Some(suffix_operator) = self
            .token
            .0
            .as_ref()
            .ok()
            .map(Token::as_suffix_operator)
            .unwrap_or(None)
        {
            let precedence = suffix_operator.precedence();
            if precedence <= lowest {
                break;
            }
            let span = self.token.1;
            bump!(self);
            match suffix_operator {
                SuffixOperator::Binary(binary_operator) => {
                    self = self.parse_infix_expr(precedence)?;
                    self.context
                        .apply_operator((Operator::Binary(binary_operator), span))
                }
                SuffixOperator::FnCall => self = self.parse_fn_call(span)?,
            }
        }
        Ok(self)
    }

    fn parse_fn_call(mut self, left: S) -> ParserResult<Self, Ctx, S> {
        let mut args = 0;
        while let Ok(token) = &self.token.0 {
            match token {
                Token::Simple(SimpleToken::RParen) => break,
                Token::Simple(SimpleToken::Comma) => {
                    bump!(self);
                    self = self.parse_fn_arg(&mut args)?;
                }
                _ => self = self.parse_fn_arg(&mut args)?,
            }
        }
        let span = self.context.diagnostics().merge_spans(&left, &self.token.1);
        self.context.apply_operator((Operator::FnCall(args), span));
        bump!(self);
        Ok(self)
    }

    fn parse_fn_arg(mut self, args: &mut usize) -> ParserResult<Self, Ctx, S> {
        self = self.parse_expression()?;
        *args += 1;
        Ok(self)
    }

    fn parse_primary_expr(mut self) -> ParserResult<Self, Ctx, S> {
        match self.token {
            (Ok(Token::Simple(LParen)), span) => {
                bump!(self);
                self.parse_parenthesized_expression(span)
            }
            _ => self.parse_atomic_expr(),
        }
    }

    fn parse_atomic_expr(mut self) -> ParserResult<Self, Ctx, S> {
        match self.token.0 {
            Ok(Token::Simple(Eof)) | Ok(Token::Simple(Eol)) => {
                Err((self, ExprParsingError::NothingParsed))
            }
            Ok(Token::Ident(ident)) => {
                self.context
                    .push_atom((ExprAtom::Ident(ident), self.token.1));
                bump!(self);
                Ok(self)
            }
            Ok(Token::Literal(literal)) => {
                self.context
                    .push_atom((ExprAtom::Literal(literal), self.token.1));
                bump!(self);
                Ok(self)
            }
            Ok(Token::Simple(SimpleToken::Dot)) => {
                self.context
                    .push_atom((ExprAtom::LocationCounter, self.token.1));
                bump!(self);
                Ok(self)
            }
            _ => {
                let span = self.token.1;
                let stripped = self.context.diagnostics().strip_span(&span);
                bump!(self);
                Err((
                    self,
                    ExprParsingError::Other(
                        Message::UnexpectedToken { token: stripped }.at(span).into(),
                    ),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{SimpleToken::*, Token::*};

    use crate::syntax::parser::mock::*;
    use crate::syntax::FinalContext;

    #[test]
    fn parse_long_sum_arg() {
        let tokens = input_tokens![
            w @ Ident(()),
            plus1 @ Plus,
            x @ Ident(()),
            plus2 @ Plus,
            y @ Literal(()),
            plus3 @ Plus,
            z @ Ident(()),
        ];
        let expected = expr()
            .ident("w")
            .ident("x")
            .plus("plus1")
            .literal("y")
            .plus("plus2")
            .ident("z")
            .plus("plus3");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_subtraction() {
        let tokens = input_tokens![x @ Ident(()), minus @ Minus, y @ Literal(())];
        let expected = expr().ident("x").literal("y").minus("minus");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_division() {
        let tokens = input_tokens![x @ Ident(()), slash @ Slash, y @ Literal(())];
        let expected = expr().ident("x").literal("y").divide("slash");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn multiplication_precedes_addition() {
        let tokens = input_tokens![
            a @ Literal(()),
            plus @ Plus,
            b @ Literal(()),
            star @ Star,
            c @ Literal(()),
        ];
        let expected = expr()
            .literal("a")
            .literal("b")
            .literal("c")
            .multiply("star")
            .plus("plus");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_sum_of_terms() {
        let tokens = input_tokens![
            a @ Literal(()),
            slash @ Slash,
            b @ Literal(()),
            plus @ Plus,
            c @ Literal(()),
            star @ Star,
            d @ Ident(()),
        ];
        let expected = expr()
            .literal("a")
            .literal("b")
            .divide("slash")
            .literal("c")
            .ident("d")
            .multiply("star")
            .plus("plus");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_multiplication() {
        let tokens = input_tokens![x @ Ident(()), star @ Star, y @ Literal(())];
        let expected = expr().ident("x").literal("y").multiply("star");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_bitwise_or() {
        let tokens = input_tokens![x @ Ident(()), pipe @ Pipe, y @ Literal(())];
        let expected = expr().ident("x").literal("y").bitwise_or("pipe");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn addition_precedes_bitwise_or() {
        let tokens =
            input_tokens![x @ Ident(()), pipe @ Pipe, y @ Ident(()), plus @ Plus, z @ Ident(())];
        let expected = expr()
            .ident("x")
            .ident("y")
            .ident("z")
            .plus("plus")
            .bitwise_or("pipe");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_nullary_fn_call() {
        let tokens = input_tokens![name @ Ident(()), left @ LParen, right @ RParen];
        let expected = expr().ident("name").fn_call(
            0,
            SymSpan::merge(TokenRef::from("left"), TokenRef::from("right")),
        );
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_unary_fn_call() {
        let tokens = input_tokens![
            name @ Ident(()),
            left @ LParen,
            arg @ Ident(()),
            right @ RParen
        ];
        let expected = expr().ident("name").ident("arg").fn_call(
            1,
            SymSpan::merge(TokenRef::from("left"), TokenRef::from("right")),
        );
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_binary_fn_call() {
        let tokens = input_tokens![
            name @ Ident(()),
            left @ LParen,
            arg1 @ Ident(()),
            Simple(Comma),
            arg2 @ Ident(()),
            right @ RParen
        ];
        let expected = expr().ident("name").ident("arg1").ident("arg2").fn_call(
            2,
            SymSpan::merge(TokenRef::from("left"), TokenRef::from("right")),
        );
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_fn_call_plus_literal() {
        let tokens = input_tokens![
            name @ Ident(()),
            left @ LParen,
            right @ RParen,
            plus @ Simple(Plus),
            literal @ Literal(())
        ];
        let expected = expr()
            .ident("name")
            .fn_call(
                0,
                SymSpan::merge(TokenRef::from("left"), TokenRef::from("right")),
            )
            .literal("literal")
            .plus("plus");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn fn_call_precedes_multiplication() {
        let tokens = input_tokens![
            literal @ Literal(()),
            star @ Simple(Star),
            name @ Ident(()),
            left @ LParen,
            right @ RParen,
        ];
        let expected = expr()
            .literal("literal")
            .ident("name")
            .fn_call(
                0,
                SymSpan::merge(TokenRef::from("left"), TokenRef::from("right")),
            )
            .multiply("star");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_location_counter() {
        let tokens = input_tokens![dot @ Simple(Dot)];
        let expected = expr().location_counter("dot");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn parse_sum_with_parentheses() {
        let tokens = input_tokens![
            a @ Ident(()),
            plus1 @ Plus,
            left @ LParen,
            b @ Ident(()),
            plus2 @ Plus,
            c @ Ident(()),
            right @ RParen,
        ];
        let expected = expr()
            .ident("a")
            .ident("b")
            .ident("c")
            .plus("plus2")
            .parentheses("left", "right")
            .plus("plus1");
        assert_eq_rpn_expr(tokens, expected)
    }

    #[test]
    fn diagnose_eof_for_rhs_operand() {
        assert_eq_rpn_expr(
            input_tokens![Ident(()), Plus],
            expr()
                .ident(0)
                .error(Message::UnexpectedEof, TokenRef::from(2)),
        )
    }

    #[test]
    fn diagnose_unexpected_token_in_expr() {
        let input = input_tokens![plus @ Plus];
        let span: SymSpan = TokenRef::from("plus").into();
        assert_eq_expr_diagnostics(
            input,
            Message::UnexpectedToken {
                token: span.clone(),
            }
            .at(span)
            .into(),
        )
    }

    fn assert_eq_rpn_expr(mut input: InputTokens, SymExpr(expected): SymExpr) {
        let parsed_rpn_expr = parse_sym_expr(&mut input);
        assert_eq!(parsed_rpn_expr, expected);
    }

    fn assert_eq_expr_diagnostics(
        mut input: InputTokens,
        expected: CompactDiagnostic<SymSpan, SymSpan>,
    ) {
        let expr_actions = parse_sym_expr(&mut input);
        assert_eq!(expr_actions, [ExprAction::EmitDiagnostic(expected)])
    }

    fn parse_sym_expr(input: &mut InputTokens) -> Vec<ExprAction<SymSpan>> {
        let tokens = &mut with_spans(&input.tokens);
        Parser::new(tokens, ExprActionCollector::new(()))
            .parse()
            .change_context(FinalContext::exit)
            .context
    }
}