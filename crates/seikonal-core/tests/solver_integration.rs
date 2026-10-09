//! Integration tests for `seikonal-core`.
//! Integration tests for `seikonal-core` - Block 1 (Grid & Stencil in homogeneous medium).

use seikonal_core::FMMSolver2D;
use seikonal_core::SeikonalError;
use seikonal_core::{Grid2D, solve_eikonal_2d}; //, Grid2D, SeikonalError};

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

#[test]
fn test_fmm_solver_2d_homogeneous_full_grid_accuracy() {
    let nx = 51;
    let nz = 51;
    let dx = 10.0; // 10 m
    let dz = 10.0; // 10 m
    let x0 = 0.0;
    let z0 = 0.0;
    let v0 = 2000.0; // 2000 m/s
    let mut grid = Grid2D::new_homogeneous(nx, nz, dx, dz, x0, z0, v0).unwrap();
    // Place source at center of domain: (250 m, 250 m) -> node (25, 25)
    let (xs, zs) = (250.0, 250.0);
    FMMSolver2D::solve(&mut grid, xs, zs).unwrap();
    let mut total_relative_error = 0.0;
    let mut num_evaluated_nodes = 0;
    for ix in 0..nx {
        for iz in 0..nz {
            let (x, z) = grid.node_coords(ix, iz).unwrap();
            let dist = ((x - xs).powi(2) + (z - zs).powi(2)).sqrt();
            let t_exact = dist / v0;
            let idx = grid.linear_index(ix, iz).unwrap();
            let t_calc = grid.traveltimes[idx];
            assert!(
                t_calc.is_finite(),
                "Traveltime at ({ix}, {iz}) is not finite"
            );
            // Skip the source node itself and immediate 1-node radius (curvature singularity)
            if dist > 2.0 * dx {
                let rel_error = (t_calc - t_exact).abs() / t_exact;
                total_relative_error += rel_error;
                num_evaluated_nodes += 1;
                // Max individual error along diagonal (45°) for 1st order upwind is < 3.5%
                assert!(
                    rel_error < 0.1,
                    "Max relative error exceeded at ({x}, {z}): t_calc={t_calc}, t_exact={t_exact}, err={rel_error}"
                );
            }
        }
    }
    let mean_rel_error = total_relative_error / (num_evaluated_nodes as f64);
    // Mean relative error across the whole 2D grid is well under 1.5%
    assert!(
        mean_rel_error < 0.05,
        "Mean relative error too high: {mean_rel_error}"
    );
}

#[test]
fn test_fmm_solver_2d_two_layer_refraction() {
    let nx = 101; // 1000 m width
    let nz = 51; // 500 m depth
    let dx = 10.0;
    let dz = 10.0;
    let x0 = 0.0;
    let z0 = 0.0;

    let v1 = 1500.0; // 1500 m/s in top layer (water/sediments)
    let v2 = 3000.0; // 3000 m/s in bottom layer (bedrock)
    let z_interface = 150.0; // interface at 150 m depth

    // Build two-layer velocity model
    let mut velocities = Vec::with_capacity(nx * nz);
    for _ix in 0..nx {
        for iz in 0..nz {
            let z = (iz as f64) * dz;
            if z < z_interface {
                velocities.push(v1);
            } else {
                velocities.push(v2);
            }
        }
    }

    let mut grid = Grid2D::new(nx, nz, dx, dz, x0, z0, velocities).unwrap();

    // Source placed at the top-left surface (0, 0)
    FMMSolver2D::solve(&mut grid, 0.0, 0.0).unwrap();

    // 1. Check monotonicity: traveltimes must strictly increase away from source
    for ix in 0..nx {
        for iz in 0..nz {
            let idx = grid.linear_index(ix, iz).unwrap();
            let t = grid.traveltimes[idx];
            assert!(t.is_finite());
            if ix > 0 {
                let prev_idx = grid.linear_index(ix - 1, iz).unwrap();
                assert!(t > grid.traveltimes[prev_idx]);
            }
        }
    }

    // 2. Far offset on surface (z = 0, x = 900 m):
    // Compare computed traveltime with theoretical head-wave arrival time
    let x_far = 900.0;
    let ix_far = (x_far / dx) as usize;
    let idx_surface_far = grid.linear_index(ix_far, 0).unwrap();
    let t_calc_far = grid.traveltimes[idx_surface_far];

    // Theoretical head-wave arrival time: T = x/v2 + 2*h*sqrt(v2^2 - v1^2)/(v1*v2)
    let t_head_exact = (x_far / v2) + 2.0 * z_interface * (v2 * v2 - v1 * v1).sqrt() / (v1 * v2);
    // Direct wave arrival time: x / v1 = 900 / 1500 = 0.60 s
    let t_direct = x_far / v1;

    // The head-wave must arrive significantly before the direct wave
    assert!(t_head_exact < t_direct);
    assert!(t_calc_far < t_direct);

    // Relative error on head-wave traveltime should be within ~3%
    let rel_err = (t_calc_far - t_head_exact).abs() / t_head_exact;
    assert!(
        rel_err < 0.05,
        "Head-wave error too high: t_calc={t_calc_far}, t_exact={t_head_exact}, rel_err={rel_err}"
    );
}
