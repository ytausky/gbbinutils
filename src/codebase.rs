use std::{cmp, ops, rc::Rc};

#[derive(Clone, Copy)]
struct BufPosition(usize);

struct BufRange {
    start: BufPosition,
    end: BufPosition,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct LineIndex(usize);

impl ops::AddAssign<usize> for LineIndex {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LineNumber(usize);

impl From<LineIndex> for LineNumber {
    fn from(line_index: LineIndex) -> LineNumber {
        LineNumber(line_index.0 + 1)
    }
}

#[derive(Debug, PartialEq)]
struct TextPosition {
    line: LineIndex,
    column_index: usize,
}

#[derive(Debug, PartialEq)]
struct TextRange {
    start: TextPosition,
    end: TextPosition,
}

trait TextBuf {
    fn text_range(&self, buf_range: &BufRange) -> TextRange;
}

struct StringSrcBuf {
    src: Rc<str>,
    line_ranges: Vec<ops::Range<usize>>,
}

impl StringSrcBuf {
    fn new(src: String) -> StringSrcBuf {
        let line_ranges = build_line_ranges(&src);
        StringSrcBuf {
            src: src.into(),
            line_ranges,
        }
    }

    fn line_index(&self, BufPosition(buf_position): BufPosition) -> LineIndex {
        match self.line_ranges
            .binary_search_by(|&ops::Range { start, end }| {
                if start <= buf_position {
                    if buf_position <= end {
                        cmp::Ordering::Equal
                    } else {
                        cmp::Ordering::Greater
                    }
                } else {
                    cmp::Ordering::Less
                }
            }) {
            Ok(line_index) => LineIndex(line_index),
            Err(_n) => panic!(),
        }
    }

    fn text_position(&self, buf_position: BufPosition) -> TextPosition {
        let line = self.line_index(buf_position);
        let line_range = &self.line_ranges[line.0];
        TextPosition {
            line,
            column_index: buf_position.0 - line_range.start,
        }
    }

    #[cfg(test)]
    fn lines<'a>(&'a self, line_range: ops::Range<LineIndex>) -> TextLines<'a> {
        TextLines {
            buf: self,
            remaining_range: line_range,
        }
    }
}

#[cfg(test)]
struct TextLines<'a> {
    buf: &'a StringSrcBuf,
    remaining_range: ops::Range<LineIndex>,
}

#[cfg(test)]
impl<'a> Iterator for TextLines<'a> {
    type Item = (LineNumber, &'a str);
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_range.start < self.remaining_range.end {
            let line_index = self.remaining_range.start;
            let line_number = line_index.into();
            let line_text = &self.buf.src[self.buf.line_ranges[line_index.0].clone()];
            self.remaining_range.start += 1;
            Some((line_number, line_text))
        } else {
            None
        }
    }
}

impl TextBuf for StringSrcBuf {
    fn text_range(&self, buf_range: &BufRange) -> TextRange {
        TextRange {
            start: self.text_position(buf_range.start),
            end: self.text_position(buf_range.end),
        }
    }
}

pub struct StringCodebase {
    bufs: Vec<StringSrcBuf>,
}

#[derive(Clone, Copy)]
pub struct BufId(usize);

impl StringCodebase {
    pub fn new() -> StringCodebase {
        StringCodebase { bufs: Vec::new() }
    }

    pub fn add_src_buf(&mut self, src: String) -> BufId {
        let buf_id = BufId(self.bufs.len());
        self.bufs.push(StringSrcBuf::new(src));
        buf_id
    }

    pub fn buf(&self, buf_id: BufId) -> Rc<str> {
        self.bufs[buf_id.0].src.clone()
    }

    #[cfg(test)]
    fn get_line(&self, buf_id: BufId, line_index: usize) -> &str {
        let buf = &self.bufs[buf_id.0];
        let &ops::Range { start, end } = &buf.line_ranges[line_index];
        &buf.src[start..end]
    }
}

fn build_line_ranges(src: &str) -> Vec<ops::Range<usize>> {
    let mut line_ranges = Vec::new();
    let mut current_line_start = 0;
    for index in src.char_indices()
        .filter(|&(_, ch)| ch == '\n')
        .map(|(index, _)| index)
    {
        line_ranges.push(ops::Range {
            start: current_line_start,
            end: index,
        });
        current_line_start = index + '\n'.len_utf8()
    }
    line_ranges.push(ops::Range {
        start: current_line_start,
        end: src.len(),
    });
    line_ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iterate_src() {
        let mut codebase = StringCodebase::new();
        let src = "src";
        let buf_id = codebase.add_src_buf(String::from(src));
        let rc_src = codebase.buf(buf_id);
        let mut iter = rc_src.char_indices();
        assert_eq!(iter.next(), Some((0, 's')));
        assert_eq!(iter.next(), Some((1, 'r')));
        assert_eq!(iter.next(), Some((2, 'c')));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn get_line() {
        let mut codebase = StringCodebase::new();
        let src = "first line\nsecond line\nthird line";
        let buf_id = codebase.add_src_buf(src.into());
        assert_eq!(codebase.get_line(buf_id, 1), "second line")
    }

    #[test]
    fn text_range_in_middle_of_line() {
        let src = "abcdefg\nhijklmn";
        let buf = StringSrcBuf::new(src.into());
        let buf_range = BufRange {
            start: BufPosition(9),
            end: BufPosition(12),
        };
        let text_range = buf.text_range(&buf_range);
        assert_eq!(
            text_range,
            TextRange {
                start: TextPosition {
                    line: LineIndex(1),
                    column_index: 1,
                },
                end: TextPosition {
                    line: LineIndex(1),
                    column_index: 4,
                },
            }
        )
    }

    #[test]
    fn borrow_some_lines() {
        let text = "my first line\nsome second line\nand a third";
        let buf = StringSrcBuf::new(text.to_string());
        let lines = buf.lines(LineIndex(1)..LineIndex(3));
        assert_eq!(
            lines.collect::<Vec<_>>(),
            [
                (LineNumber(2), "some second line"),
                (LineNumber(3), "and a third")
            ]
        )
    }
}
