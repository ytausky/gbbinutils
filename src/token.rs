#[derive(Clone, Debug, PartialEq)]
pub enum Token<'a> {
    Word(&'a str),
    Eol,
}
