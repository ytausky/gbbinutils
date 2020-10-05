use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

pub struct Log<T>(Rc<RefCell<Vec<T>>>);

impl<T> Log<T> {
    pub fn new() -> Self {
        Log(Rc::new(RefCell::new(Vec::new())))
    }

    pub fn into_inner(self) -> Vec<T>
    where
        T: Debug,
    {
        Rc::try_unwrap(self.0).unwrap().into_inner()
    }

    pub fn push(&self, datum: impl Into<T>) {
        self.0.borrow_mut().push(datum.into())
    }

    pub fn clear(&self) {
        self.0.borrow_mut().clear()
    }
}

impl<T> Clone for Log<T> {
    fn clone(&self) -> Self {
        Log(Rc::clone(&self.0))
    }
}

pub fn with_log<T: Debug>(f: impl FnOnce(Log<T>)) -> Vec<T> {
    let log = Log::new();
    f(log.clone());
    log.into_inner()
}
