use std::fmt;
use std::iter::FusedIterator;
use std::mem::ManuallyDrop;
use std::ops::{Index, IndexMut};
use std::ptr::NonNull;

pub struct GslVec(NonNull<gsl_sys::gsl_vector>);

impl GslVec {
    pub fn new(len: usize) -> Self {
        let ptr = unsafe { gsl_sys::gsl_vector_calloc(len) };
        let ptr = NonNull::new(ptr).expect("out of memory");
        Self(ptr)
    }

    pub fn len(&self) -> usize {
        self.as_ref().size
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<&f64> {
        if index < self.len() {
            Some(unsafe { &*gsl_sys::gsl_vector_const_ptr(self.as_raw(), index) })
        } else {
            None
        }
    }

    pub fn get_mut(&self, index: usize) -> Option<&mut f64> {
        if index < self.len() {
            Some(unsafe { &mut *gsl_sys::gsl_vector_ptr(self.as_raw(), index) })
        } else {
            None
        }
    }

    pub fn fill(&mut self, x: f64) {
        unsafe { gsl_sys::gsl_vector_set_all(self.as_raw(), x) }
    }

    pub fn iter(&self) -> GslVecIter<'_> {
        GslVecIter { vec: self, curr: 0 }
    }

    pub fn as_slice(&self) -> &[f64] {
        unsafe { as_slice(self.as_raw()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        unsafe { as_mut_slice(self.as_raw()) }
    }

    pub fn as_raw(&self) -> *mut gsl_sys::gsl_vector {
        self.0.as_ptr()
    }

    pub fn into_raw(v: Self) -> *mut gsl_sys::gsl_vector {
        ManuallyDrop::new(v).as_raw()
    }

    /// # Safety
    ///
    /// Given pointer must be a valid allocation of GSL vector produced either
    /// by [`into_raw`](Self::into_raw) or by an external call to
    /// `gsl_vector_alloc` or `gsl_vector_calloc`.
    pub unsafe fn from_raw(raw: *mut gsl_sys::gsl_vector) -> Self {
        let ptr = NonNull::new(raw).expect("invalid pointer");
        Self(ptr)
    }

    fn as_ref(&self) -> &gsl_sys::gsl_vector {
        unsafe { self.0.as_ref() }
    }
}

pub(crate) unsafe fn as_slice<'a>(ptr: *const gsl_sys::gsl_vector) -> &'a [f64] {
    std::slice::from_raw_parts((*ptr).data, (*ptr).size)
}

pub(crate) unsafe fn as_mut_slice<'a>(ptr: *mut gsl_sys::gsl_vector) -> &'a mut [f64] {
    std::slice::from_raw_parts_mut((*ptr).data, (*ptr).size)
}

impl Drop for GslVec {
    fn drop(&mut self) {
        unsafe { gsl_sys::gsl_vector_free(self.as_raw()) }
    }
}

impl From<&[f64]> for GslVec {
    fn from(s: &[f64]) -> Self {
        let ptr = unsafe { gsl_sys::gsl_vector_alloc(s.len()) };
        let ptr = NonNull::new(ptr).expect("out of memory");

        for (i, v) in s.iter().copied().enumerate() {
            unsafe {
                gsl_sys::gsl_vector_set(ptr.as_ptr(), i, v);
            }
        }

        Self(ptr)
    }
}

impl From<GslVec> for Vec<f64> {
    fn from(v: GslVec) -> Self {
        v.iter().copied().collect()
    }
}

impl From<Vec<f64>> for GslVec {
    fn from(v: Vec<f64>) -> Self {
        Self::from(v.as_slice())
    }
}

impl Index<usize> for GslVec {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl IndexMut<usize> for GslVec {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("index out of bounds")
    }
}

impl Clone for GslVec {
    fn clone(&self) -> Self {
        let cloned = Self::new(self.len());

        unsafe {
            gsl_sys::gsl_vector_memcpy(cloned.as_raw(), self.as_raw());
        }

        cloned
    }
}

impl PartialEq for GslVec {
    fn eq(&self, other: &Self) -> bool {
        unsafe { gsl_sys::gsl_vector_equal(self.as_raw(), other.as_raw()) == 1 }
    }
}

impl Eq for GslVec {}

impl fmt::Debug for GslVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GslVec([")?;

        let mut it = self.iter();

        if let Some(v) = it.next() {
            write!(f, "{}", v)?;
        }

        for v in it {
            write!(f, ", {}", v)?;
        }

        write!(f, "])")
    }
}

// SAFETY: GslVec follows standard rules for aliasing. Thus, Sync invariants are
// checked by Rust borrow checker. Send is satisfied, because GslVec does not
// implement Copy and its Clone implementation does not share a single
// allocation.
unsafe impl Send for GslVec {}
unsafe impl Sync for GslVec {}

pub struct GslVecIter<'a> {
    vec: &'a GslVec,
    curr: usize,
}

impl GslVecIter<'_> {
    fn remaining(&self) -> usize {
        self.vec.len() - self.curr
    }

    fn consume(&mut self) {
        self.curr = self.vec.len()
    }
}

impl<'a> Iterator for GslVecIter<'a> {
    type Item = &'a f64;

    fn next(&mut self) -> Option<Self::Item> {
        let curr = self.curr;
        self.curr = std::cmp::min(curr + 1, self.vec.len());
        self.vec.get(curr)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.remaining();
        (r, Some(r))
    }

    fn count(mut self) -> usize
    where
        Self: Sized,
    {
        let r = self.remaining();
        self.consume();
        r
    }

    fn last(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        if !self.vec.is_empty() {
            let last = &self.vec[self.vec.len() - 1];
            self.consume();
            Some(last)
        } else {
            None
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.curr += n;
        self.vec.get(self.curr)
    }
}

impl ExactSizeIterator for GslVecIter<'_> {}
impl FusedIterator for GslVecIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut x = GslVec::new(3);
        assert_eq!(x.len(), 3);
        assert_eq!(x[0], 0.0);

        x[0] = 3.0;
        assert_eq!(x[0], 3.0);
    }

    #[test]
    fn slice() {
        let x = GslVec::from(&[3.0, 2.0, 1.0][..]);
        assert_eq!(x.as_slice(), &[3.0, 2.0, 1.0]);
    }

    #[test]
    fn fill() {
        let mut x = GslVec::new(3);
        x.fill(3.0);
        assert_eq!(x.as_slice(), &[3.0, 3.0, 3.0]);
    }

    #[test]
    fn iter() {
        let x = GslVec::from(&[3.0, 2.0, 1.0][..]);

        let mut it = x.iter();
        assert_eq!(it.next(), Some(&3.0));
        assert_eq!(it.next(), Some(&2.0));
        assert_eq!(it.next(), Some(&1.0));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn clone() {
        let x = GslVec::from(&[3.0, 2.0, 1.0][..]);
        let y = x.clone();
        assert_ne!(x.as_raw(), y.as_raw());
    }

    #[test]
    fn equality() {
        let x = GslVec::from(&[3.0, 2.0, 1.0][..]);
        let mut y = x.clone();
        assert_eq!(x, y);

        y[0] = 42.0;
        assert_ne!(x, y);
    }
}
