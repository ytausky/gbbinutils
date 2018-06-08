use instruction::{Direction, RelocExpr};
use Width;

#[derive(Debug, PartialEq)]
pub enum Node<SR> {
    Byte(u8),
    Expr(RelocExpr<SR>, Width),
    LdInlineAddr(RelocExpr<SR>, Direction),
}
