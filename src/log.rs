use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

pub struct Log<T>(Rc<RefCell<Vec<T>>>);

impl<T> Default for Log<T> {
    fn default() -> Self {
        Self(Rc::new(RefCell::new(Vec::new())))
    }
}

impl<T> Log<T> {
    pub fn into_inner(self) -> Vec<T>
    where
        T: Debug,
    {
        Rc::try_unwrap(self.0).unwrap().into_inner()
    }

    pub fn push(&self, datum: impl Into<T>) {
        self.0.borrow_mut().push(datum.into())
    }
}

impl<T> Clone for Log<T> {
    fn clone(&self) -> Self {
        Log(Rc::clone(&self.0))
    }
}
