#[derive(Clone, Debug, PartialEq)]
pub enum Token<'a> {
    Comma,
    Eol,
    QuotedString(&'a str),
    Word(&'a str),
}
