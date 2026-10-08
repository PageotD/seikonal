//! 2D Cartesian grid representation and index mapping for the Seikonal solver.
//!
//! Conventions:
//! - The grid is defined by its dimensions (nx, nz) and spacings (dx, dz).
//! - The grid points are indexed in a row-major order, where the z-direction is the fast axis and the x-direction is the slow axis.
//! - The z-axis is positive downwards, and x-axis is positive to the right.
//! - Flat memory storage in C-order: `index = ix * nz + iz`.

use crate::error::SeikonalError;

/// 2D regular cartesian grid for eikonal solving.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid2D {
    /// Number of grid nodes along the x-axis (slow axis)
    pub nx: usize,
    /// Number of grid nodes along the z-axis (fast axis, vertical, positive downwards)
    pub nz: usize,
    /// Grid spacing along the x-axis (>0)
    pub dx: f64,
    /// Grid spacing along the z-axis (>0)
    pub dz: f64,
    /// Origin of the x coordinate in physical space
    pub x0: f64,
    /// Origin of the z coordinate (depth at top) in physical space
    pub z0: f64,
    /// Velocity field values in flat C-order (nx*nz)
    pub velocities: Vec<f64>,
    /// Traveltime field values in flat C-order (nx*nz)
    pub traveltimes: Vec<f64>,
}

impl Grid2D {
    /// Creates a new `Grid2D` instance with uniform velocity.
    pub fn new_homogeneous(
        nx: usize,
        nz: usize,
        dx: f64,
        dz: f64,
        x0: f64,
        z0: f64,
        velocity: f64,
    ) -> Result<Self, SeikonalError> {
        if nx == 0 || nz == 0 {
            return Err(SeikonalError::InvalidDimensions { nx, nz });
        }
        if dx <= 0.0 || dz <= 0.0 {
            return Err(SeikonalError::InvalidSpacing { dx, dz });
        }
        if velocity <= 0.0 {
            return Err(SeikonalError::InvalidVelocity { velocity, index: 0 });
        }

        let num_nodes = nx * nz;
        Ok(Self {
            nx,
            nz,
            dx,
            dz,
            x0,
            z0,
            velocities: vec![velocity; num_nodes],
            traveltimes: vec![f64::INFINITY; num_nodes],
        })
    }

    /// Creates a new `Grid2D` with a given velocity field.
    pub fn new(
        nx: usize,
        nz: usize,
        dx: f64,
        dz: f64,
        x0: f64,
        z0: f64,
        velocities: Vec<f64>,
    ) -> Result<Self, SeikonalError> {
        if nx == 0 || nz == 0 {
            return Err(SeikonalError::InvalidDimensions { nx, nz });
        }
        if dx <= 0.0 || dz <= 0.0 {
            return Err(SeikonalError::InvalidSpacing { dx, dz });
        }
        let num_nodes = nx * nz;
        if velocities.len() != num_nodes {
            return Err(SeikonalError::InvalidVelocity {
                velocity: 0.0,
                index: velocities.len(),
            });
        }

        for (i, &v) in velocities.iter().enumerate() {
            if v <= 0.0 {
                return Err(SeikonalError::InvalidVelocity {
                    velocity: v,
                    index: i,
                });
            }
        }

        Ok(Self {
            nx,
            nz,
            dx,
            dz,
            x0,
            z0,
            velocities,
            traveltimes: vec![f64::INFINITY; num_nodes],
        })
    }

    /// Converts 2D grids coordinates (ix, iz) in a linear 1D index (C-order).
    #[inline]
    pub fn linear_index(&self, ix: usize, iz: usize) -> Result<usize, SeikonalError> {
        if ix >= self.nx || iz >= self.nz {
            return Err(SeikonalError::IndexOutOfBounds {
                index: (ix, iz),
                dimensions: (self.nx, self.nz),
            });
        }
        Ok(ix * self.nz + iz)
    }

    /// Converts a linear 1D index back to 2D grid coordinates (ix, iz)
    #[inline]
    pub fn grid_coords(&self, linear_index: usize) -> Result<(usize, usize), SeikonalError> {
        if linear_index >= self.nx * self.nz {
            return Err(SeikonalError::IndexOutOfBounds {
                index: (linear_index, 0),
                dimensions: (self.nx, self.nz),
            });
        }
        let ix = linear_index / self.nz;
        let iz = linear_index % self.nz;
        Ok((ix, iz))
    }

    /// Compute continuous physical coordinates (x, z) for node (ix, iz).
    #[inline]
    pub fn node_coords(&self, ix: usize, iz: usize) -> Result<(f64, f64), SeikonalError> {
        if ix >= self.nx || iz >= self.nz {
            return Err(SeikonalError::IndexOutOfBounds {
                index: (ix, iz),
                dimensions: (self.nx, self.nz),
            });
        }
        let x = self.x0 + (ix as f64) * self.dx;
        let z = self.z0 + (iz as f64) * self.dz;
        Ok((x, z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_grid_creation_and_index_mapping() {
        let nx = 4;
        let nz = 3;
        let dx = 10.0;
        let dz = 5.0;
        let grid = Grid2D::new_homogeneous(nx, nz, dx, dz, 0.0, 0.0, 2000.0).unwrap();
        assert_eq!(grid.velocities.len(), 12);
        assert_eq!(grid.traveltimes.len(), 12);
        assert!(grid.traveltimes[0].is_infinite());
        // Test C-order mapping: ix=2, iz=1 => index = 2 * 3 + 1 = 7
        let idx = grid.linear_index(2, 1).unwrap();
        assert_eq!(idx, 7);
        let (ix, iz) = grid.grid_coords(7).unwrap();
        assert_eq!((ix, iz), (2, 1));
        // Physical coordinates: ix=2 => x=20.0, iz=1 => z=5.0
        let (x, z) = grid.node_coords(2, 1).unwrap();
        assert_eq!((x, z), (20.0, 5.0));
    }
    #[test]
    fn test_grid_invalid_inputs() {
        // Zero dimensions
        assert!(Grid2D::new_homogeneous(0, 5, 1.0, 1.0, 0.0, 0.0, 1000.0).is_err());
        // Non-positive spacing
        assert!(Grid2D::new_homogeneous(5, 5, -1.0, 1.0, 0.0, 0.0, 1000.0).is_err());
        // Non-positive velocity
        assert!(Grid2D::new_homogeneous(5, 5, 1.0, 1.0, 0.0, 0.0, 0.0).is_err());
        // Out of bounds
        let grid = Grid2D::new_homogeneous(3, 3, 1.0, 1.0, 0.0, 0.0, 1000.0).unwrap();
        assert!(grid.linear_index(3, 0).is_err());
    }
}
