//! Fast Marching Method (FFM) eikonal solver

use crate::error::SeikonalError;
use crate::grid::Grid2D;
use crate::narrowband::{NarrowBand, NodeState};
use crate::stencil::solve_eikonal_2d;

/// 2D FFM eikonal solver.
pub struct FMMSolver2D;

impl FMMSolver2D {
    /// Solves the eikonal equation on a 2D grid for a single point source (xs, zs).
    ///
    /// # Arguments
    /// - `grid`: mutable referenceto the 2D grid (traveltimes updated in place)
    /// - `xs`: x coordiante of the source point
    /// - `zs`: z coordinate of the source point (depth)
    ///
    /// # Returns
    /// `Ok(())` on success, or `SeikonalError`
    pub fn solve(grid: &mut Grid2D, xs: f64, zs: f64) -> Result<(), SeikonalError> {
        let num_nodes = grid.nx * grid.nz;
        let mut nb = NarrowBand::new(num_nodes);

        // Initialize traveltimes
        grid.traveltimes.fill(f64::INFINITY);

        // Initialize source node
        Self::initialize_point_source(grid, &mut nb, xs, zs)?;

        // Main loop
        while let Some((idx, _t_min)) = nb.pop_min(&grid.traveltimes) {
            nb.set_state(idx, NodeState::Alive);

            let (ix, iz) = grid.grid_coords(idx)?;

            // Check neighbors
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

                // Canddadate traveltime at neighbor node
                let t_candidate = Self::compute_node_traveltime(grid, &nb, n_ix, n_iz)?;

                if t_candidate < grid.traveltimes[n_idx] {
                    grid.traveltimes[n_idx] = t_candidate;
                    nb.push(n_idx, t_candidate)
                }
            }
        }

        Ok(())
    }

    /// Initialise nearest grid node to the source with local analytical traveltime
    fn initialize_point_source(
        grid: &mut Grid2D,
        nb: &mut NarrowBand,
        xs: f64,
        zs: f64,
    ) -> Result<(), SeikonalError> {
        let ix_s = (((xs - grid.x0) / grid.dx).round() as isize).max(0) as usize;
        let iz_s = (((zs - grid.z0) / grid.dz).round() as isize).max(0) as usize;
        let ix_s = ix_s.min(grid.nx - 1);
        let iz_s = iz_s.min(grid.nz - 1);
        let idx_s = grid.linear_index(ix_s, iz_s)?;
        let (x_node, z_node) = grid.node_coords(ix_s, iz_s)?;
        let dist = ((x_node - xs).powi(2) + (z_node - zs).powi(2)).sqrt();
        let v0 = grid.velocities[idx_s];
        let t0 = dist / v0;
        grid.traveltimes[idx_s] = t0;
        nb.push(idx_s, t0);
        Ok(())
    }

    /// Computes traveltime at (ix, iz) from known `Alive` neighbors.
    fn compute_node_traveltime(
        grid: &Grid2D,
        nb: &NarrowBand,
        ix: usize,
        iz: usize,
    ) -> Result<f64, SeikonalError> {
        let idx = grid.linear_index(ix, iz)?;
        let v = grid.velocities[idx];
        // Find min traveltime from Alive neighbors along X
        let mut tx_min = None;
        if ix > 0 {
            let left_idx = grid.linear_index(ix - 1, iz)?;
            if nb.state(left_idx) == NodeState::Alive {
                tx_min = Some(grid.traveltimes[left_idx]);
            }
        }
        if ix + 1 < grid.nx {
            let right_idx = grid.linear_index(ix + 1, iz)?;
            if nb.state(right_idx) == NodeState::Alive {
                let tr = grid.traveltimes[right_idx];
                tx_min = Some(tx_min.map_or(tr, |tx| tx.min(tr)));
            }
        }
        // Find min traveltime from Alive neighbors along Z
        let mut tz_min = None;
        if iz > 0 {
            let top_idx = grid.linear_index(ix, iz - 1)?;
            if nb.state(top_idx) == NodeState::Alive {
                tz_min = Some(grid.traveltimes[top_idx]);
            }
        }
        if iz + 1 < grid.nz {
            let bottom_idx = grid.linear_index(ix, iz + 1)?;
            if nb.state(bottom_idx) == NodeState::Alive {
                let tb = grid.traveltimes[bottom_idx];
                tz_min = Some(tz_min.map_or(tb, |tz| tz.min(tb)));
            }
        }
        Ok(solve_eikonal_2d(tx_min, tz_min, grid.dx, grid.dz, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_solver_homogeneous_small_grid() {
        let mut grid = Grid2D::new_homogeneous(5, 5, 10.0, 10.0, 0.0, 0.0, 2000.0).unwrap();
        // Source placed exactly at corner node (0, 0)
        FMMSolver2D::solve(&mut grid, 0.0, 0.0).unwrap();
        // Node (0, 0) should have traveltime = 0.0
        let idx_00 = grid.linear_index(0, 0).unwrap();
        assert_eq!(grid.traveltimes[idx_00], 0.0);
        // Node (1, 0) should have traveltime = 10.0 / 2000.0 = 0.005 s
        let idx_10 = grid.linear_index(1, 0).unwrap();
        assert!((grid.traveltimes[idx_10] - 0.005).abs() < 1e-12);
        // Node (0, 1) should have traveltime = 10.0 / 2000.0 = 0.005 s
        let idx_01 = grid.linear_index(0, 1).unwrap();
        assert!((grid.traveltimes[idx_01] - 0.005).abs() < 1e-12);
        // All nodes must have finite traveltime
        assert!(grid.traveltimes.iter().all(|&t| t.is_finite()));
    }
}
