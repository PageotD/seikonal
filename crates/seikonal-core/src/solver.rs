//! Complete Fast Marching Method (FMM) eikonal solver in 2D.

use crate::error::SeikonalError;
use crate::grid::Grid2D;
use crate::narrowband::{NarrowBand, NodeState};
use crate::stencil::solve_eikonal_2d_order;

/// 2D Fast Marching Method eikonal solver.
pub struct FMMSolver2D;

impl FMMSolver2D {
    /// Solves the Eikonal equation on a 2D grid for a single continuous point source (xs, zs)
    /// using 1st order upwind scheme by default.
    #[inline]
    pub fn solve(grid: &mut Grid2D, xs: f64, zs: f64) -> Result<(), SeikonalError> {
        Self::solve_order(grid, xs, zs, 1)
    }

    /// Solves the Eikonal equation on a 2D grid with a chosen stencil order (1 or 2).
    ///
    /// # Arguments
    /// - `grid`: Mutable reference to the 2D grid.
    /// - `xs`: X coordinate of the source point.
    /// - `zs`: Z coordinate of the source point (depth).
    /// - `order`: Upwind finite difference order (1 or 2).
    pub fn solve_order(
        grid: &mut Grid2D,
        xs: f64,
        zs: f64,
        order: usize,
    ) -> Result<(), SeikonalError> {
        let num_nodes = grid.nx * grid.nz;
        let mut nb = NarrowBand::new(num_nodes);

        // Reset all traveltimes to infinity
        grid.traveltimes.fill(f64::INFINITY);

        // 1. Initialize source neighborhood
        Self::initialize_point_source(grid, &mut nb, xs, zs, order)?;

        // 2. Main Fast Marching loop
        while let Some((idx, _t_min)) = nb.pop_min(&grid.traveltimes) {
            nb.set_state(idx, NodeState::Alive);

            let (ix, iz) = grid.grid_coords(idx)?;

            // Check 4 direct neighbors
            let neighbors = [
                (ix.wrapping_sub(1), iz, ix > 0),
                (ix + 1, iz, ix + 1 < grid.nx),
                (ix, iz.wrapping_sub(1), iz > 0),
                (ix, iz + 1, iz + 1 < grid.nz),
            ];

            for &(n_ix, n_iz, is_valid) in &neighbors {
                if !is_valid {
                    continue;
                }

                let n_idx = grid.linear_index(n_ix, n_iz)?;
                if nb.state(n_idx) == NodeState::Alive {
                    continue;
                }

                let t_cand = Self::compute_node_traveltime(grid, &nb, n_ix, n_iz, order)?;

                if t_cand < grid.traveltimes[n_idx] {
                    grid.traveltimes[n_idx] = t_cand;
                    nb.push(n_idx, t_cand);
                }
            }
        }

        Ok(())
    }

    /// Initializes source neighborhood with exact analytical traveltimes and seeds the narrow band.
    fn initialize_point_source(
        grid: &mut Grid2D,
        nb: &mut NarrowBand,
        xs: f64,
        zs: f64,
        order: usize,
    ) -> Result<(), SeikonalError> {
        let ix_center = (((xs - grid.x0) / grid.dx).round() as isize).max(0) as usize;
        let iz_center = (((zs - grid.z0) / grid.dz).round() as isize).max(0) as usize;

        let ix_center = ix_center.min(grid.nx - 1);
        let iz_center = iz_center.min(grid.nz - 1);

        // Freeze 3x3 box as Alive with exact analytical traveltimes
        let ix_min = ix_center.saturating_sub(1);
        let ix_max = (ix_center + 1).min(grid.nx - 1);
        let iz_min = iz_center.saturating_sub(1);
        let iz_max = (iz_center + 1).min(grid.nz - 1);

        for ix in ix_min..=ix_max {
            for iz in iz_min..=iz_max {
                let idx = grid.linear_index(ix, iz)?;
                let (x, z) = grid.node_coords(ix, iz)?;
                let dist = ((x - xs).powi(2) + (z - zs).powi(2)).sqrt();
                let v = grid.velocities[idx];
                let t_exact = dist / v;

                grid.traveltimes[idx] = t_exact;
                nb.set_state(idx, NodeState::Alive);
            }
        }

        // Initialize outer rim nodes
        for ix in ix_min..=ix_max {
            for iz in iz_min..=iz_max {
                let neighbors = [
                    (ix.wrapping_sub(1), iz, ix > 0),
                    (ix + 1, iz, ix + 1 < grid.nx),
                    (ix, iz.wrapping_sub(1), iz > 0),
                    (ix, iz + 1, iz + 1 < grid.nz),
                ];

                for &(n_ix, n_iz, is_valid) in &neighbors {
                    if !is_valid {
                        continue;
                    }
                    let n_idx = grid.linear_index(n_ix, n_iz)?;
                    if nb.state(n_idx) == NodeState::Far {
                        let t_cand = Self::compute_node_traveltime(grid, nb, n_ix, n_iz, order)?;
                        if t_cand < grid.traveltimes[n_idx] {
                            grid.traveltimes[n_idx] = t_cand;
                            nb.push(n_idx, t_cand);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Computes candidate traveltime at (ix, iz) from known Alive neighbors using chosen stencil order.
    fn compute_node_traveltime(
        grid: &Grid2D,
        nb: &NarrowBand,
        ix: usize,
        iz: usize,
        order: usize,
    ) -> Result<f64, SeikonalError> {
        let idx = grid.linear_index(ix, iz)?;
        let v = grid.velocities[idx];

        // 1. Analyze X axis: find min Alive neighbor and its second neighbor
        let (tx1, tx2) = Self::get_axis_neighbors(grid, nb, ix, iz, true)?;

        // 2. Analyze Z axis: find min Alive neighbor and its second neighbor
        let (tz1, tz2) = Self::get_axis_neighbors(grid, nb, ix, iz, false)?;

        Ok(solve_eikonal_2d_order(
            tx1, tx2, tz1, tz2, grid.dx, grid.dz, v, order,
        ))
    }

    /// Retrieves first and second Alive neighbor traveltimes along an axis (X or Z).
    fn get_axis_neighbors(
        grid: &Grid2D,
        nb: &NarrowBand,
        ix: usize,
        iz: usize,
        is_x_axis: bool,
    ) -> Result<(Option<f64>, Option<f64>), SeikonalError> {
        let (pos, len) = if is_x_axis {
            (ix, grid.nx)
        } else {
            (iz, grid.nz)
        };

        let t_neg = if pos > 0 {
            let n_idx = if is_x_axis {
                grid.linear_index(ix - 1, iz)?
            } else {
                grid.linear_index(ix, iz - 1)?
            };
            if nb.state(n_idx) == NodeState::Alive {
                Some(grid.traveltimes[n_idx])
            } else {
                None
            }
        } else {
            None
        };

        let t_pos = if pos + 1 < len {
            let n_idx = if is_x_axis {
                grid.linear_index(ix + 1, iz)?
            } else {
                grid.linear_index(ix, iz + 1)?
            };
            if nb.state(n_idx) == NodeState::Alive {
                Some(grid.traveltimes[n_idx])
            } else {
                None
            }
        } else {
            None
        };

        match (t_neg, t_pos) {
            (Some(t_n), Some(t_p)) => {
                if t_n <= t_p {
                    let t2 = Self::get_second_neighbor(grid, nb, ix, iz, is_x_axis, -1)?;
                    Ok((Some(t_n), t2))
                } else {
                    let t2 = Self::get_second_neighbor(grid, nb, ix, iz, is_x_axis, 1)?;
                    Ok((Some(t_p), t2))
                }
            }
            (Some(t_n), None) => {
                let t2 = Self::get_second_neighbor(grid, nb, ix, iz, is_x_axis, -1)?;
                Ok((Some(t_n), t2))
            }
            (None, Some(t_p)) => {
                let t2 = Self::get_second_neighbor(grid, nb, ix, iz, is_x_axis, 1)?;
                Ok((Some(t_p), t2))
            }
            (None, None) => Ok((None, None)),
        }
    }

    /// Fetches the second neighbor along the chosen direction (-1 or +1).
    fn get_second_neighbor(
        grid: &Grid2D,
        nb: &NarrowBand,
        ix: usize,
        iz: usize,
        is_x_axis: bool,
        dir: isize,
    ) -> Result<Option<f64>, SeikonalError> {
        let (n2_ix, n2_iz, valid) = if is_x_axis {
            let target_ix = ix as isize + 2 * dir;
            (
                target_ix.max(0) as usize,
                iz,
                target_ix >= 0 && (target_ix as usize) < grid.nx,
            )
        } else {
            let target_iz = iz as isize + 2 * dir;
            (
                ix,
                target_iz.max(0) as usize,
                target_iz >= 0 && (target_iz as usize) < grid.nz,
            )
        };

        if valid {
            let idx2 = grid.linear_index(n2_ix, n2_iz)?;
            if nb.state(idx2) == NodeState::Alive {
                return Ok(Some(grid.traveltimes[idx2]));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_homogeneous_small_grid() {
        let mut grid = Grid2D::new_homogeneous(5, 5, 10.0, 10.0, 0.0, 0.0, 2000.0).unwrap();
        FMMSolver2D::solve(&mut grid, 0.0, 0.0).unwrap();

        let idx_00 = grid.linear_index(0, 0).unwrap();
        assert_eq!(grid.traveltimes[idx_00], 0.0);
        assert!(grid.traveltimes.iter().all(|&t| t.is_finite()));
    }
}
