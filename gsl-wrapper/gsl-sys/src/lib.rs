#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// https://github.com/rust-lang/rust/issues/54341
#![allow(improper_ctypes)]

include!(concat!(env!("OUT_DIR"), "/gsl_sys.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn vectors() {
        unsafe {
            let x = gsl_vector_alloc(3);
            assert_ne!(x, ptr::null_mut());

            gsl_vector_set(x, 0, 3.14);
            assert_eq!(gsl_vector_get(x, 0), 3.14);

            gsl_vector_free(x);
        }
    }
}
