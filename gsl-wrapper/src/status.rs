use std::ffi::CStr;
use std::fmt;

#[derive(Clone, Copy, PartialEq)]
pub struct GslStatus(i32);

impl GslStatus {
    pub fn ok() -> Self {
        Self(0)
    }

    pub fn err(error: GslError) -> Self {
        Self(error.to_int())
    }

    pub fn as_str(&self) -> &str {
        // SAFETY: The pointer returned from `gsl_strerror` points to a static
        // string.
        let cstr = unsafe { CStr::from_ptr(gsl_sys::gsl_strerror(self.0)) };
        cstr.to_str().unwrap()
    }

    pub fn is_ok(&self) -> bool {
        self.0 == 0
    }

    pub fn is_err(&self) -> bool {
        !self.is_ok()
    }

    pub fn to_result(self) -> Result<(), String> {
        if self.is_ok() {
            Ok(())
        } else {
            Err(self.as_str().to_string())
        }
    }

    pub(crate) fn from_raw(s: i32) -> Self {
        Self(s)
    }

    pub fn to_raw(self) -> i32 {
        self.0
    }
}

impl fmt::Debug for GslStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GslStatus({:?})", self.as_str())
    }
}

impl From<GslError> for GslStatus {
    fn from(error: GslError) -> Self {
        Self(error.to_int())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GslError {
    BadFunc,
    Singular,
    NoProgress,
    Continue,
}

impl GslError {
    fn to_int(self) -> i32 {
        match self {
            GslError::BadFunc => gsl_sys::GSL_EBADFUNC,
            GslError::Singular => gsl_sys::GSL_ESING,
            GslError::NoProgress => gsl_sys::GSL_ENOPROG,
            GslError::Continue => gsl_sys::GSL_CONTINUE,
        }
    }
}

impl PartialEq<GslError> for GslStatus {
    fn eq(&self, other: &GslError) -> bool {
        self.0 == other.to_int()
    }
}

impl PartialEq<GslStatus> for GslError {
    fn eq(&self, other: &GslStatus) -> bool {
        other == self
    }
}
