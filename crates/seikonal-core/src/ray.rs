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
