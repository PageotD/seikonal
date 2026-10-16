//! Python bindings for `seikonal`.

use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use seikonal_core::{FMMSolver2D, Grid2D, SeikonalError, trace_ray_2d};

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
#[pyfunction]
#[pyo3(signature = (velocities, dx, dz, xs, zs, x0 = 0.0, z0 = 0.0, order = 2))]
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
    order: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if order != 1 && order != 2 {
        return Err(PyValueError::new_err("order must be 1 or 2"));
    }

    let shape = velocities.shape();
    let nx = shape[0];
    let nz = shape[1];

    if !velocities.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "velocities array must be C-contiguous (C-order)",
        ));
    }

    let vel_slice = velocities
        .as_slice()
        .map_err(|_| PyValueError::new_err("velocities array must be C-contiguous float64"))?;

    let vel_vec = vel_slice.to_vec();

    let mut grid = Grid2D::new(nx, nz, dx, dz, x0, z0, vel_vec).map_err(to_py_err)?;
    FMMSolver2D::solve_order(&mut grid, xs, zs, order).map_err(to_py_err)?;

    let tt_vec = grid.traveltimes;
    let py_array = tt_vec.into_pyarray(py);
    let py_array_2d = py_array
        .reshape([nx, nz])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(py_array_2d)
}

/// Helper function to build a Grid2D wrapper containing existing traveltimes.
fn build_grid_from_traveltimes(
    traveltimes: &PyReadonlyArray2<'_, f64>,
    dx: f64,
    dz: f64,
    x0: f64,
    z0: f64,
) -> PyResult<Grid2D> {
    let shape = traveltimes.shape();
    let nx = shape[0];
    let nz = shape[1];

    if !traveltimes.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "traveltimes array must be C-contiguous (C-order)",
        ));
    }

    let tt_slice = traveltimes
        .as_slice()
        .map_err(|_| PyValueError::new_err("traveltimes array must be C-contiguous float64"))?;

    let mut grid = Grid2D::new_homogeneous(nx, nz, dx, dz, x0, z0, 1.0).map_err(to_py_err)?;
    grid.traveltimes = tt_slice.to_vec();
    Ok(grid)
}

/// Traces a single seismic ray from receiver (xr, zr) back to source (xs, zs).
///
/// Parameters
/// ----------
/// traveltimes : numpy.ndarray (2D, float64, C-contiguous)
///     Computed traveltime field with shape (nx, nz).
/// dx : float
///     Grid spacing along X.
/// dz : float
///     Grid spacing along Z.
/// xs : float
///     Source X coordinate.
/// zs : float
///     Source Z coordinate (depth).
/// xr : float
///     Receiver X coordinate.
/// zr : float
///     Receiver Z coordinate (depth).
/// x0 : float, optional
///     Grid origin X (default: 0.0).
/// z0 : float, optional
///     Grid origin Z (default: 0.0).
/// step_size : float, optional
///     Ray integration step size. Defaults to `0.5 * min(dx, dz)`.
///
/// Returns
/// -------
/// numpy.ndarray
///     2D array of coordinates along the ray of shape (N, 2), ordered from receiver to source.
#[pyfunction]
#[pyo3(signature = (traveltimes, dx, dz, xs, zs, xr, zr, x0 = 0.0, z0 = 0.0, step_size = None))]
#[allow(clippy::too_many_arguments)]
fn trace_ray<'py>(
    py: Python<'py>,
    traveltimes: PyReadonlyArray2<'py, f64>,
    dx: f64,
    dz: f64,
    xs: f64,
    zs: f64,
    xr: f64,
    zr: f64,
    x0: f64,
    z0: f64,
    step_size: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let grid = build_grid_from_traveltimes(&traveltimes, dx, dz, x0, z0)?;
    let ray = trace_ray_2d(&grid, xs, zs, xr, zr, step_size, None).map_err(to_py_err)?;

    let num_points = ray.points.len();
    let mut flat_coords = Vec::with_capacity(num_points * 2);
    for (x, z) in ray.points {
        flat_coords.push(x);
        flat_coords.push(z);
    }

    let py_arr = flat_coords.into_pyarray(py);
    let py_arr_2d = py_arr
        .reshape([num_points, 2])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(py_arr_2d)
}

/// Traces multiple seismic rays from an array of receivers to a source.
///
/// Parameters
/// ----------
/// traveltimes : numpy.ndarray (2D, float64, C-contiguous)
///     Computed traveltime field with shape (nx, nz).
/// dx : float
///     Grid spacing along X.
/// dz : float
///     Grid spacing along Z.
/// xs : float
///     Source X coordinate.
/// zs : float
///     Source Z coordinate.
/// receivers : numpy.ndarray (2D, float64)
///     Receiver coordinates array of shape (M, 2) where each row is `(xr, zr)`.
/// x0 : float, optional
///     Grid origin X (default: 0.0).
/// z0 : float, optional
///     Grid origin Z (default: 0.0).
/// step_size : float, optional
///     Ray integration step size. Defaults to `0.5 * min(dx, dz)`.
///
/// Returns
/// -------
/// list of numpy.ndarray
///     List of ray trajectories, each of shape (N_i, 2).
#[pyfunction]
#[pyo3(signature = (traveltimes, dx, dz, xs, zs, receivers, x0 = 0.0, z0 = 0.0, step_size = None))]
#[allow(clippy::too_many_arguments)]
fn trace_rays<'py>(
    py: Python<'py>,
    traveltimes: PyReadonlyArray2<'py, f64>,
    dx: f64,
    dz: f64,
    xs: f64,
    zs: f64,
    receivers: PyReadonlyArray2<'py, f64>,
    x0: f64,
    z0: f64,
    step_size: Option<f64>,
) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
    let rec_shape = receivers.shape();
    if rec_shape.len() != 2 || rec_shape[1] != 2 {
        return Err(PyValueError::new_err("receivers must have shape (M, 2)"));
    }

    let grid = build_grid_from_traveltimes(&traveltimes, dx, dz, x0, z0)?;
    let rec_slice = receivers
        .as_slice()
        .map_err(|_| PyValueError::new_err("receivers array must be C-contiguous float64"))?;

    let num_receivers = rec_shape[0];
    let mut ray_list = Vec::with_capacity(num_receivers);

    for m in 0..num_receivers {
        let xr = rec_slice[2 * m];
        let zr = rec_slice[2 * m + 1];

        let ray = trace_ray_2d(&grid, xs, zs, xr, zr, step_size, None).map_err(to_py_err)?;
        let n_pts = ray.points.len();
        let mut flat = Vec::with_capacity(n_pts * 2);
        for (x, z) in ray.points {
            flat.push(x);
            flat.push(z);
        }
        let py_arr = flat.into_pyarray(py);
        let py_2d = py_arr
            .reshape([n_pts, 2])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        ray_list.push(py_2d);
    }

    Ok(ray_list)
}

/// The `seikonal` Python module.
#[pymodule]
fn seikonal(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_2d, m)?)?;
    m.add_function(wrap_pyfunction!(trace_ray, m)?)?;
    m.add_function(wrap_pyfunction!(trace_rays, m)?)?;
    Ok(())
}
