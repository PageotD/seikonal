//! Error types for `seikonal-core`.

use std::fmt;

/// Derive the `Error` trait for a type.
#[derive(Debug, Clone, PartialEq)]

pub enum SeikonalError {
    /// Invalid grid dimensions
    InvalidDimensions {
        /// Number of grid points in the x-direction
        nx: usize,
        /// Number of grid points in the z-direction
        nz: usize,
    },
    /// Invalid grid spacing
    InvalidSpacing {
        /// Grid spacing in the x-direction
        dx: f64,
        /// Grid spacing in the z-direction
        dz: f64,
    },
    /// Invalid velocity value
    InvalidVelocity {
        /// Velocity value at a specific grid point
        velocity: f64,
        /// Linear index of the grid point with the invalid velocity
        index: usize,
    },
    /// Grid index out of bounds
    IndexOutOfBounds {
        /// Linear index of the grid point that is out of bounds
        index: (usize, usize),
        /// Dimensions of the grid (nx, nz)
        dimensions: (usize, usize),
    },
}

impl fmt::Display for SeikonalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { nx, nz } => {
                write!(
                    f,
                    "Invalid grid dimensions ({nx}, {nz}): all dimensions must be >=1"
                )
            }
            Self::InvalidSpacing { dx, dz } => {
                write!(
                    f,
                    "Invalid grid spacing ({dx}, {dz}): all spacings must be >0"
                )
            }
            Self::InvalidVelocity { velocity, index } => {
                write!(
                    f,
                    "Invalid velocity {velocity} at linear index {index}: velocity must be >0"
                )
            }
            Self::IndexOutOfBounds { index, dimensions } => {
                write!(
                    f,
                    "Index ({}, {}) is out of bounds for dimensions ({}, {})",
                    index.0, index.1, dimensions.0, dimensions.1
                )
            }
        }
    }
}

/// Implement the `std::error::Error` trait for `SeikonalError`.
/// This allows `SeikonalError` to be used with the standard error handling mechanisms in Rust.
impl std::error::Error for SeikonalError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = SeikonalError::InvalidDimensions { nx: 0, nz: 0 };
        assert!(err.to_string().contains("Invalid grid dimensions"));

        let err = SeikonalError::InvalidSpacing { dx: 0.0, dz: 0.0 };
        assert!(err.to_string().contains("Invalid grid spacing"));

        let err = SeikonalError::InvalidVelocity {
            velocity: -1.0,
            index: 0,
        };
        assert!(err.to_string().contains("Invalid velocity"));

        let err = SeikonalError::IndexOutOfBounds {
            index: (5, 5),
            dimensions: (3, 3),
        };
        assert!(err.to_string().contains("is out of bounds"));
    }
}
