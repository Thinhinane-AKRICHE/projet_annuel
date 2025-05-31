mod linear_model;
use linear_model::LinearModel;

use std::ffi::c_void;

#[no_mangle]
pub extern "C" fn create_linear_model(n_features: usize, lr: f64, epochs: usize) -> *mut c_void {
    let model = Box::new(LinearModel::new(n_features, lr, epochs));
    Box::into_raw(model) as *mut c_void
}

#[no_mangle]
pub extern "C" fn train_linear_model(
    model_ptr: *mut c_void,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
) {
    let model = unsafe { &mut *(model_ptr as *mut LinearModel) };
    let X = unsafe { std::slice::from_raw_parts(x_ptr, n_samples * n_features) };
    let Y = unsafe { std::slice::from_raw_parts(y_ptr, n_samples) };

    let X_rows: Vec<Vec<f64>> = X.chunks(n_features).map(|c| c.to_vec()).collect();
    let Y_vec = Y.to_vec();

    model.fit(&X_rows, &Y_vec, None, None); // régression ici (activation = None)
}

#[no_mangle]
pub extern "C" fn predict_linear_model(model_ptr: *mut c_void, x_ptr: *const f64, n_features: usize) -> f64 {
    let model = unsafe { &mut *(model_ptr as *mut LinearModel) };
    let x = unsafe { std::slice::from_raw_parts(x_ptr, n_features) };
    model.predict(&x.to_vec(), None)
}
