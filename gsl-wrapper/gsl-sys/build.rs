use std::env;
use std::path::PathBuf;

fn main() {
    // Link GSL and BLAS shared libraries.
    println!("cargo:rustc-link-lib=gsl");
    println!("cargo:rustc-link-lib=gslcblas");

    // Generate only if the warpper.h changed.
    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .size_t_is_usize(true)
        // Be very strict in inclusion.
        .whitelist_function("gsl_vector_.*")
        .whitelist_function("gsl_multiroot_fsolver_.*")
        .whitelist_var("gsl_multiroot_fsolver_.*")
        .whitelist_function("gsl_multiroot_test_.*")
        .whitelist_function("gsl_strerror")
        .whitelist_var("GSL_E.*")
        .whitelist_var("GSL_SUCCESS")
        .whitelist_var("GSL_CONTINUE")
        .blacklist_function(".*_(fread|fwrite|fscanf|fprintf)")
        .blacklist_item("stream|IO|FILE")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("gsl_sys.rs"))
        .expect("Unable to write bindings");
}
