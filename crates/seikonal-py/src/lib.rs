//! Python bindings for `seikonal`.

use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use seikonal_core::{FMMSolver2D, Grid2D, SeikonalError};

/// Converts internal `SeikonalError` into an appropriate Python exception.
fn to_py_err(err: SeikonalError) -> PyErr {
    match err {
        SeikonalError::InvalidDimensions { .. }
        | SeikonalError::InvalidSpacing { .. }
        | SeikonalError::InvalidVelocity { .. } => PyValueError::new_err(err.to_string()),
        SeikonalError::IndexOutOfBounds { .. } => PyIndexError::new_err(err.to_string()),
    }
}

/// Solves the 2D Eikonal equation on a Cartesian grid.
///
/// Parameters
/// ----------
/// velocities : numpy.ndarray (2D, float64, C-contiguous)
///     Velocity model with shape (nx, nz).
/// dx : float
///     Grid spacing along X axis (> 0.0).
/// dz : float
///     Grid spacing along Z axis (> 0.0).
/// xs : float
///     Source X coordinate.
/// zs : float
///     Source Z coordinate (depth).
/// x0 : float, optional
///     Origin X coordinate (default: 0.0).
/// z0 : float, optional
///     Origin Z coordinate (default: 0.0).
///
/// Returns
/// -------
/// numpy.ndarray (2D, float64)
///     Computed traveltime field with shape (nx, nz).
#[pyfunction]
#[pyo3(signature = (velocities, dx, dz, xs, zs, x0 = 0.0, z0 = 0.0))]
#[allow(clippy::too_many_arguments)]
fn solve_2d<'py>(
    py: Python<'py>,
    velocities: PyReadonlyArray2<'py, f64>,
    dx: f64,
    dz: f64,
    xs: f64,
    zs: f64,
    x0: f64,
    z0: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = velocities.shape();
    let nx = shape[0];
    let nz = shape[1];

    // Ensure C-contiguous flat slice
    let vel_slice = velocities
        .as_slice()
        .map_err(|_| PyValueError::new_err("velocities array must be C-contiguous float64"))?;

    let vel_vec = vel_slice.to_vec();

    // Create Grid2D
    let mut grid = Grid2D::new(nx, nz, dx, dz, x0, z0, vel_vec).map_err(to_py_err)?;

    // Solve FMM
    FMMSolver2D::solve(&mut grid, xs, zs).map_err(to_py_err)?;

    // Convert traveltimes Vec<f64> back into a 2D NumPy array with shape (nx, nz)
    let tt_vec = grid.traveltimes;
    let py_array = tt_vec.into_pyarray(py);
    let py_array_2d = py_array
        .reshape([nx, nz])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(py_array_2d)
}

/// The `seikonal` Python module.
#[pymodule]
fn seikonal(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_2d, m)?)?;
    Ok(())
}
