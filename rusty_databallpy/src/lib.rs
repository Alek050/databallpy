use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::PyReadonlyArray2;
use ndarray::Array2;
use std::collections::HashMap;
mod needleman_wunch;
mod read_tracab_tracking_data;

#[pyfunction]
fn rusty_needleman_wunch(
    sim_mat: PyReadonlyArray2<f32>,
    gap_event: f32,
    gap_frame: f32,
) -> PyResult<HashMap<i32, i32>> {
    let sim_mat: Array2<f32> = sim_mat.as_array().to_owned();
    let solved_needleman_wunch = needleman_wunch::needleman_wunch(sim_mat, gap_event, gap_frame);

    Ok(solved_needleman_wunch)
}

#[pyfunction]
fn rusty_read_tracab_txt_data(tracab_file_loc: &str) -> PyResult<()> {
    read_tracab_tracking_data::save_tracab_dat_to_parquet(tracab_file_loc)?;
    Ok(())
}


#[pymodule]
fn rusty_databallpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rusty_needleman_wunch, m)?)?;
    m.add_function(wrap_pyfunction!(rusty_read_tracab_txt_data, m)?)?;
    Ok(())
}
