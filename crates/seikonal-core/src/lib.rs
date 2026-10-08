//! `seikonal-core` provides the core Fast Marching Method (FMM) eikonal solver algorithms.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod grid;
pub mod stencil;

pub use error::SeikonalError;
pub use grid::Grid2D;
pub use stencil::solve_eikonal_2d;
