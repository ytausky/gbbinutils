#[derive(Clone, Debug, PartialEq)]
pub enum Token<'a> {
    Comma,
    Eol,
    Number(isize),
    QuotedString(&'a str),
    Word(&'a str),
}
