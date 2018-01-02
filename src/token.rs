#[derive(PartialEq)]
pub enum Token<'a> {
    Word(&'a str),
    Eol,
}
