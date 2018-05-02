use std::{cmp, fs, ops, rc::Rc};

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct LineIndex(usize);

impl ops::Add<usize> for LineIndex {
    type Output = LineIndex;
    fn add(mut self, rhs: usize) -> Self::Output {
        self += rhs;
        self
    }
}

impl ops::AddAssign<usize> for LineIndex {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineNumber(pub usize);

impl From<LineIndex> for LineNumber {
    fn from(line_index: LineIndex) -> LineNumber {
        LineNumber(line_index.0 + 1)
    }
}

#[derive(Debug, PartialEq)]
pub struct TextPosition {
    pub line: LineIndex,
    column_index: usize,
}

#[derive(Debug, PartialEq)]
pub struct TextRange {
    pub start: TextPosition,
    pub end: TextPosition,
}

pub type BufRange = ops::Range<usize>;

pub trait TextBuf {
    fn text_range(&self, buf_range: &BufRange) -> TextRange;
}

pub struct StringSrcBuf {
    src: Rc<str>,
    line_ranges: Vec<BufRange>,
}

impl StringSrcBuf {
    fn new(src: String) -> StringSrcBuf {
        let line_ranges = build_line_ranges(&src);
        StringSrcBuf {
            src: src.into(),
            line_ranges,
        }
    }

    fn line_index(&self, buf_offset: usize) -> LineIndex {
        match self.line_ranges
            .binary_search_by(|&ops::Range { start, end }| {
                if start <= buf_offset {
                    if buf_offset <= end {
                        cmp::Ordering::Equal
                    } else {
                        cmp::Ordering::Less
                    }
                } else {
                    cmp::Ordering::Greater
                }
            }) {
            Ok(line_index) => LineIndex(line_index),
            Err(_n) => panic!("couldn't find buffer position {}", buf_offset),
        }
    }

    fn text_position(&self, buf_offset: usize) -> TextPosition {
        let line = self.line_index(buf_offset);
        let line_range = &self.line_ranges[line.0];
        TextPosition {
            line,
            column_index: buf_offset - line_range.start,
        }
    }

    #[cfg(test)]
    pub fn lines<'a>(&'a self, line_range: ops::Range<LineIndex>) -> TextLines<'a> {
        TextLines {
            buf: self,
            remaining_range: line_range,
        }
    }

    pub fn text(&self) -> Rc<str> {
        self.src.clone()
    }
}

#[cfg(test)]
pub struct TextLines<'a> {
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

pub struct TextCache {
    bufs: Vec<StringSrcBuf>,
}

#[derive(Clone, Copy, Debug)]
pub struct BufId(usize);

impl TextCache {
    pub fn new() -> TextCache {
        TextCache { bufs: Vec::new() }
    }

    pub fn add_src_buf(&mut self, src: String) -> BufId {
        let buf_id = BufId(self.bufs.len());
        self.bufs.push(StringSrcBuf::new(src));
        buf_id
    }

    pub fn buf(&self, buf_id: BufId) -> &StringSrcBuf {
        &self.bufs[buf_id.0]
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

pub trait FileSystem {
    fn read_file(&mut self, filename: &str) -> String;
}

pub struct StdFileSystem;

impl StdFileSystem {
    pub fn new() -> StdFileSystem {
        StdFileSystem {}
    }
}

impl FileSystem for StdFileSystem {
    fn read_file(&mut self, filename: &str) -> String {
        use std::io::prelude::*;
        let mut file = fs::File::open(filename).unwrap();
        let mut src = String::new();
        file.read_to_string(&mut src).unwrap();
        src
    }
}

pub trait Codebase {
    fn open(&mut self, path: &str) -> BufId;
    fn buf(&self, buf_id: BufId) -> Rc<str>;
}

pub struct FileCodebase<FS: FileSystem> {
    fs: FS,
    cache: TextCache,
}

impl<FS: FileSystem> FileCodebase<FS> {
    pub fn new(fs: FS) -> FileCodebase<FS> {
        FileCodebase {
            fs,
            cache: TextCache::new(),
        }
    }
}

impl<FS: FileSystem> Codebase for FileCodebase<FS> {
    fn open(&mut self, path: &str) -> BufId {
        let text = self.fs.read_file(path);
        self.cache.add_src_buf(text)
    }

    fn buf(&self, buf_id: BufId) -> Rc<str> {
        self.cache.buf(buf_id).text()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iterate_src() {
        let mut cache = TextCache::new();
        let src = "src";
        let buf_id = cache.add_src_buf(String::from(src));
        let rc_src = cache.buf(buf_id).text();
        let mut iter = rc_src.char_indices();
        assert_eq!(iter.next(), Some((0, 's')));
        assert_eq!(iter.next(), Some((1, 'r')));
        assert_eq!(iter.next(), Some((2, 'c')));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn get_line() {
        let mut cache = TextCache::new();
        let src = "first line\nsecond line\nthird line";
        let buf_id = cache.add_src_buf(src.into());
        assert_eq!(cache.get_line(buf_id, 1), "second line")
    }

    #[test]
    fn text_range_in_middle_of_line() {
        let src = "abcdefg\nhijklmn";
        let buf = StringSrcBuf::new(src.into());
        let buf_range = 9..12;
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

    #[test]
    fn line_ranges() {
        let text = "    nop\n    my_macro a, $12\n\n";
        assert_eq!(build_line_ranges(text), [0..7, 8..27, 28..28, 29..29])
    }
}
