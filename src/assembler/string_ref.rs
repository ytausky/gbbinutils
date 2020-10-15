use std::borrow::Borrow;
use std::fmt::{Debug, Error, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Bound, Deref, Range, RangeBounds};
use std::rc::Rc;

#[derive(Clone, Eq)]
pub struct StringRef {
    full: Rc<str>,
    range: Range<usize>,
}

impl StringRef {
    pub fn new(full: Rc<str>, range: Range<usize>) -> Self {
        Self { full, range }
    }

    pub fn substring<R: RangeBounds<usize>>(&self, range: R) -> Self {
        let start = self.range.start
            + match range.start_bound() {
                Bound::Unbounded => 0,
                Bound::Included(n) => *n,
                Bound::Excluded(n) => *n + 1,
            };
        let end = self.range.start
            + match range.end_bound() {
                Bound::Unbounded => self.range.len(),
                Bound::Included(n) => *n + 1,
                Bound::Excluded(n) => *n,
            };
        assert!(start >= self.range.start);
        assert!(end <= self.range.end);
        Self {
            full: self.full.clone(),
            range: start..end,
        }
    }

    fn as_str(&self) -> &str {
        &self.full[self.range.clone()]
    }
}

impl AsRef<str> for StringRef {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Borrow<str> for StringRef {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl Debug for StringRef {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        self.as_str().fmt(f)
    }
}

impl Deref for StringRef {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl From<&str> for StringRef {
    fn from(string: &str) -> Self {
        Self::new(string.to_owned().into(), 0..string.len())
    }
}

impl Hash for StringRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state)
    }
}

impl PartialEq for StringRef {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}
