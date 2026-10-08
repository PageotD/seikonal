//! Integration tests for `seikonal-core`.
//! Integration tests for `seikonal-core` - Block 1 (Grid & Stencil in homogeneous medium).

use seikonal_core::SeikonalError;
use seikonal_core::{Grid2D, solve_eikonal_2d};

#[test]
fn test_integration_error_handling() {
    let err = SeikonalError::InvalidDimensions { nx: 0, nz: 10 };
    assert!(err.to_string().contains("Invalid grid dimensions"));
}

#[test]
fn test_homogeneous_2d_stencil_against_analytical() {
    let nx = 51;
    let nz = 51;
    let dx = 10.0;
    let dz = 10.0;
    let x0 = 0.0;
    let z0 = 0.0;
    let v0 = 2000.0; // 2000 m/s
    let grid = Grid2D::new_homogeneous(nx, nz, dx, dz, x0, z0, v0).unwrap();
    // Source placed at center: (250.0, 250.0)
    let (xs, zs) = (250.0, 250.0);
    let analytical_traveltime =
        |x: f64, z: f64| -> f64 { ((x - xs).powi(2) + (z - zs).powi(2)).sqrt() / v0 };
    // Evaluate stencil further away from source at (ix=40, iz=40) -> (400.0, 400.0)
    // using analytical values for upwind neighbors (39, 40) and (40, 39)
    let (x_left, z_left) = grid.node_coords(39, 40).unwrap();
    let (x_top, z_top) = grid.node_coords(40, 39).unwrap();
    let tx = analytical_traveltime(x_left, z_left);
    let tz = analytical_traveltime(x_top, z_top);
    let t_calc = solve_eikonal_2d(Some(tx), Some(tz), dx, dz, v0);
    let (x_target, z_target) = grid.node_coords(40, 40).unwrap();
    let t_exact = analytical_traveltime(x_target, z_target);
    // Further from source, relative error drops below 1%
    let rel_error = (t_calc - t_exact).abs() / t_exact;
    assert!(
        rel_error < 0.01,
        "Relative error too high: t_calc={t_calc}, t_exact={t_exact}, rel_error={rel_error}"
    );
}
#[test]
fn test_homogeneous_1d_stencil_exact() {
    let dx = 5.0;
    let dz = 5.0;
    let v0 = 1500.0;
    // Pure 1D propagation along X axis
    let t0 = 0.1;
    let t_next = solve_eikonal_2d(Some(t0), None, dx, dz, v0);
    let expected = t0 + dx / v0;
    assert!((t_next - expected).abs() < 1e-12);
}
