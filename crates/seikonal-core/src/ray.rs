//! Seismic ray tracing by gradient descent on the traveltime field.

use crate::error::SeikonalError;
use crate::grid::Grid2D;

/// Computes the discrete gradient (dT/dx, dT/dz) at node (ix, iz) using finite differences.
#[inline]
pub fn node_gradient(grid: &Grid2D, ix: usize, iz: usize) -> Result<(f64, f64), SeikonalError> {
    if ix >= grid.nx || iz >= grid.nz {
        return Err(SeikonalError::IndexOutOfBounds {
            index: (ix, iz),
            dimensions: (grid.nx, grid.nz),
        });
    }

    // dT/dx
    let dtdx = if grid.nx == 1 {
        0.0
    } else if ix == 0 {
        let t0 = grid.traveltimes[grid.linear_index(0, iz)?];
        let t1 = grid.traveltimes[grid.linear_index(1, iz)?];
        (t1 - t0) / grid.dx
    } else if ix + 1 == grid.nx {
        let t0 = grid.traveltimes[grid.linear_index(ix - 1, iz)?];
        let t1 = grid.traveltimes[grid.linear_index(ix, iz)?];
        (t1 - t0) / grid.dx
    } else {
        let t_prev = grid.traveltimes[grid.linear_index(ix - 1, iz)?];
        let t_next = grid.traveltimes[grid.linear_index(ix + 1, iz)?];
        (t_next - t_prev) / (2.0 * grid.dx)
    };

    // dT/dz
    let dtdz = if grid.nz == 1 {
        0.0
    } else if iz == 0 {
        let t0 = grid.traveltimes[grid.linear_index(ix, 0)?];
        let t1 = grid.traveltimes[grid.linear_index(ix, 1)?];
        (t1 - t0) / grid.dz
    } else if iz + 1 == grid.nz {
        let t0 = grid.traveltimes[grid.linear_index(ix, iz - 1)?];
        let t1 = grid.traveltimes[grid.linear_index(ix, iz)?];
        (t1 - t0) / grid.dz
    } else {
        let t_prev = grid.traveltimes[grid.linear_index(ix, iz - 1)?];
        let t_next = grid.traveltimes[grid.linear_index(ix, iz + 1)?];
        (t_next - t_prev) / (2.0 * grid.dz)
    };

    Ok((dtdx, dtdz))
}

/// Evaluates the gradient (dT/dx, dT/dz) at an arbitrary continuous point (x, z)
/// using bilinear interpolation of node gradients.
pub fn interpolated_gradient(
    grid: &Grid2D,
    x: f64,
    z: f64,
) -> Result<Option<(f64, f64)>, SeikonalError> {
    let x_max = grid.x0 + (grid.nx.saturating_sub(1) as f64) * grid.dx;
    let z_max = grid.z0 + (grid.nz.saturating_sub(1) as f64) * grid.dz;

    if x < grid.x0 || x > x_max || z < grid.z0 || z > z_max {
        return Ok(None);
    }

    // Grid cell indices
    let fx = ((x - grid.x0) / grid.dx).max(0.0);
    let fz = ((z - grid.z0) / grid.dz).max(0.0);

    let ix = (fx.floor() as usize).min(grid.nx.saturating_sub(2));
    let iz = (fz.floor() as usize).min(grid.nz.saturating_sub(2));

    let (x_node, z_node) = grid.node_coords(ix, iz)?;
    let rx = ((x - x_node) / grid.dx).clamp(0.0, 1.0);
    let rz = ((z - z_node) / grid.dz).clamp(0.0, 1.0);

    // Gradients at the 4 cell corners
    let g00 = node_gradient(grid, ix, iz)?;
    let g10 = node_gradient(grid, ix + 1, iz)?;
    let g01 = node_gradient(grid, ix, iz + 1)?;
    let g11 = node_gradient(grid, ix + 1, iz + 1)?;

    // Bilinear interpolation
    let dtdx = (1.0 - rx) * (1.0 - rz) * g00.0
        + rx * (1.0 - rz) * g10.0
        + (1.0 - rx) * rz * g01.0
        + rx * rz * g11.0;

    let dtdz = (1.0 - rx) * (1.0 - rz) * g00.1
        + rx * (1.0 - rz) * g10.1
        + (1.0 - rx) * rz * g01.1
        + rx * rz * g11.1;

    Ok(Some((dtdx, dtdz)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_gradient_linear_field() {
        // T(x, z) = 2.0 * x + 3.0 * z
        // Expected gradient everywhere: (2.0, 3.0)
        let nx = 5;
        let nz = 5;
        let dx = 10.0;
        let dz = 10.0;
        let mut grid = Grid2D::new_homogeneous(nx, nz, dx, dz, 0.0, 0.0, 1000.0).unwrap();

        for ix in 0..nx {
            for iz in 0..nz {
                let (x, z) = grid.node_coords(ix, iz).unwrap();
                let idx = grid.linear_index(ix, iz).unwrap();
                grid.traveltimes[idx] = 2.0 * x + 3.0 * z;
            }
        }

        // Test center node
        let (gx, gz) = node_gradient(&grid, 2, 2).unwrap();
        assert!((gx - 2.0).abs() < 1e-10);
        assert!((gz - 3.0).abs() < 1e-10);

        // Test boundary nodes
        let (gx_b, gz_b) = node_gradient(&grid, 0, 4).unwrap();
        assert!((gx_b - 2.0).abs() < 1e-10);
        assert!((gz_b - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolated_gradient_linear_field() {
        let nx = 5;
        let nz = 5;
        let dx = 10.0;
        let dz = 10.0;
        let mut grid = Grid2D::new_homogeneous(nx, nz, dx, dz, 0.0, 0.0, 1000.0).unwrap();

        for ix in 0..nx {
            for iz in 0..nz {
                let (x, z) = grid.node_coords(ix, iz).unwrap();
                let idx = grid.linear_index(ix, iz).unwrap();
                grid.traveltimes[idx] = 2.0 * x + 3.0 * z;
            }
        }

        // Any continuous point (e.g. x = 12.3, z = 34.7)
        let grad = interpolated_gradient(&grid, 12.3, 34.7).unwrap().unwrap();
        assert!((grad.0 - 2.0).abs() < 1e-10);
        assert!((grad.1 - 3.0).abs() < 1e-10);

        // Outside grid domain
        assert!(interpolated_gradient(&grid, -5.0, 10.0).unwrap().is_none());
    }
}

/// A 2D seismic ray path.
#[derive(Debug, Clone, PartialEq)]
pub struct Ray2D {
    /// Trajectory coordinates `[(x0, z0), (x1, z1), ...]`, starting from receiver to source.
    pub points: Vec<(f64, f64)>,
    /// Total integrated arclength (distance) along the ray.
    pub length: f64,
}

/// Traces a seismic ray from a receiver position (xr, zr) back to the source (xs, zs)
/// using Runge-Kutta 2nd-order (Heun) gradient descent on the traveltime field.
///
/// # Arguments
/// - `grid`: 2D grid containing the solved traveltimes.
/// - `xs`: Source X coordinate.
/// - `zs`: Source Z coordinate (depth).
/// - `xr`: Receiver X coordinate.
/// - `zr`: Receiver Z coordinate (depth).
/// - `step_size`: Integration step size `ds`. If `None`, defaults to `0.5 * min(dx, dz)`.
/// - `max_steps`: Maximum number of integration steps (default: `10 * (nx + nz)`).
pub fn trace_ray_2d(
    grid: &Grid2D,
    xs: f64,
    zs: f64,
    xr: f64,
    zr: f64,
    step_size: Option<f64>,
    max_steps: Option<usize>,
) -> Result<Ray2D, SeikonalError> {
    let ds = step_size.unwrap_or_else(|| 0.5 * grid.dx.min(grid.dz));
    let max_iter = max_steps.unwrap_or_else(|| 10 * (grid.nx + grid.nz));

    let mut points = Vec::with_capacity(128);
    points.push((xr, zr));

    let mut current_pos = (xr, zr);
    let mut total_length = 0.0;

    let get_direction = |pos: (f64, f64)| -> Result<Option<(f64, f64)>, SeikonalError> {
        if let Some((gx, gz)) = interpolated_gradient(grid, pos.0, pos.1)? {
            let norm = (gx * gx + gz * gz).sqrt();
            if norm > 1e-12 {
                // Descend opposite to gradient: -grad / |grad|
                return Ok(Some((-gx / norm, -gz / norm)));
            }
        }
        Ok(None)
    };

    for _ in 0..max_iter {
        let dist_to_source = ((current_pos.0 - xs).powi(2) + (current_pos.1 - zs).powi(2)).sqrt();

        // If within 1 step of the source, snap directly to source and finish
        if dist_to_source <= ds {
            points.push((xs, zs));
            total_length += dist_to_source;
            break;
        }

        // RK2 (Heun): k1
        let k1 = match get_direction(current_pos)? {
            Some(dir) => dir,
            None => break, // Out of bounds or flat gradient
        };

        // Predictor step
        let pred_pos = (current_pos.0 + ds * k1.0, current_pos.1 + ds * k1.1);

        // RK2 (Heun): k2
        let k2 = match get_direction(pred_pos)? {
            Some(dir) => dir,
            None => k1, // Fallback to Euler if predictor left the grid
        };

        // Corrector step
        let next_pos = (
            current_pos.0 + 0.5 * ds * (k1.0 + k2.0),
            current_pos.1 + 0.5 * ds * (k1.1 + k2.1),
        );

        let step_len =
            ((next_pos.0 - current_pos.0).powi(2) + (next_pos.1 - current_pos.1).powi(2)).sqrt();

        // Check if we are stalling or moving away from source
        let next_dist_to_source = ((next_pos.0 - xs).powi(2) + (next_pos.1 - zs).powi(2)).sqrt();
        if next_dist_to_source > dist_to_source + 1e-6 {
            // Overshot or moving away, terminate by connecting to source
            points.push((xs, zs));
            total_length += dist_to_source;
            break;
        }

        current_pos = next_pos;
        points.push(current_pos);
        total_length += step_len;
    }

    Ok(Ray2D {
        points,
        length: total_length,
    })
}
