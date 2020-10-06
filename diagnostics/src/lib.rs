use std::fmt;
use std::ops::{Add, AddAssign};

/// A full description of an assembler diagnostic.
///
/// A [`Diagnostic`](struct.Diagnostic.html) contains all the information required to display a
/// meaningful diagnostic message to a user. It consists of one or more
/// [`Clause`](struct.Clause.html)s, where the first one describes the object of the diagnostic and
/// the rest provide more context if necessary.
#[derive(Debug, PartialEq)]
pub struct Diagnostic {
    pub clauses: Vec<Clause>,
}

/// A single clause of an assembler diagnostic.
///
/// A [`Clause`](struct.Clause.html) contains a message that's relevant to a particular source file
/// or a part thereof. In addition it may include an excerpt from said file, as well as an optional
/// highlight within it.
#[derive(Debug, PartialEq)]
pub struct Clause {
    pub file: String,
    pub tag: Tag,
    pub message: String,
    pub excerpt: Option<Excerpt>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tag {
    Error,
    Note,
}

/// A code excerpt with an optional highlight.
///
/// An [`Excerpt`](struct.Excerpt.html) contains a single line of code meant to provide context for
/// a diagnostic clause. The optional highlight can pinpoint the interesting part of the line.
#[derive(Debug, PartialEq)]
pub struct Excerpt {
    pub line: LineNumber,
    pub source: String,
    pub highlight: Option<TextRange>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineNumber(pub usize);

#[derive(Debug, PartialEq)]
pub struct TextPosition {
    pub line: LineIndex,
    pub column_index: usize,
}

#[derive(Debug, PartialEq)]
pub struct TextRange {
    pub start: TextPosition,
    pub end: TextPosition,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct LineIndex(pub usize);

impl Add<usize> for LineIndex {
    type Output = LineIndex;
    fn add(mut self, rhs: usize) -> Self::Output {
        self += rhs;
        self
    }
}

impl AddAssign<usize> for LineIndex {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs
    }
}

impl From<LineIndex> for LineNumber {
    fn from(LineIndex(index): LineIndex) -> LineNumber {
        LineNumber(index + 1)
    }
}

impl From<LineNumber> for LineIndex {
    fn from(LineNumber(n): LineNumber) -> LineIndex {
        assert_ne!(n, 0);
        LineIndex(n - 1)
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for clause in &self.clauses {
            write!(f, "{}", clause)?
        }
        Ok(())
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.excerpt {
            None => writeln!(f, "{}: {}: {}", self.file, self.tag, self.message),
            Some(location) => {
                let squiggle = location
                    .highlight
                    .as_ref()
                    .map_or_else(String::new, mk_squiggle);
                writeln!(
                    f,
                    "{}:{}: {}: {}\n{}{}",
                    self.file, location.line, self.tag, self.message, location.source, squiggle,
                )
            }
        }
    }
}

impl fmt::Display for LineNumber {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.0.fmt(formatter)
    }
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Tag::Error => "error",
            Tag::Note => "note",
        })
    }
}

fn mk_squiggle(range: &TextRange) -> String {
    assert_eq!(range.start.line, range.end.line);

    use std::cmp::max;
    let space_count = range.start.column_index;
    let caret_count = max(range.end.column_index - space_count, 1);

    use std::iter::{once, repeat};
    let spaces = repeat(' ').take(space_count);
    let carets = repeat('^').take(caret_count);
    once('\n').chain(spaces).chain(carets).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    static DUMMY_FILE: &str = "/my/file";

    #[test]
    fn render_elaborated_diagnostic() {
        let diagnostic = Diagnostic {
            clauses: vec![Clause {
                file: DUMMY_FILE.to_string(),
                tag: Tag::Error,
                message: "invocation of undefined macro `my_macro`".to_string(),
                excerpt: Some(Excerpt {
                    line: LineNumber(2),
                    source: "    my_macro a, $12".to_string(),
                    highlight: mk_highlight(LineNumber(2), 4, 12),
                }),
            }],
        };
        let expected = r"/my/file:2: error: invocation of undefined macro `my_macro`
    my_macro a, $12
    ^^^^^^^^
";
        assert_eq!(diagnostic.to_string(), expected)
    }

    #[test]
    fn render_diagnostic_without_source() {
        let diagnostic = Diagnostic {
            clauses: vec![Clause {
                file: DUMMY_FILE.to_string(),
                tag: Tag::Error,
                message: "file constains invalid UTF-8".to_string(),
                excerpt: None,
            }],
        };
        let expected = r"/my/file: error: file constains invalid UTF-8
";
        assert_eq!(diagnostic.to_string(), expected);
    }

    #[test]
    fn highlight_eos_with_one_tilde() {
        let elaborated = Diagnostic {
            clauses: vec![Clause {
                file: DUMMY_FILE.to_string(),
                tag: Tag::Error,
                message: "unexpected end of file".into(),
                excerpt: Some(Excerpt {
                    line: LineNumber(2),
                    source: "dummy".to_string(),
                    highlight: mk_highlight(LineNumber(2), 5, 5),
                }),
            }],
        };
        let expected = r"/my/file:2: error: unexpected end of file
dummy
     ^
";
        assert_eq!(elaborated.to_string(), expected)
    }

    fn mk_highlight(line_number: LineNumber, start: usize, end: usize) -> Option<TextRange> {
        Some(TextRange {
            start: TextPosition {
                line: line_number.into(),
                column_index: start,
            },
            end: TextPosition {
                line: line_number.into(),
                column_index: end,
            },
        })
    }
}
