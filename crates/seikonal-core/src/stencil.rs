//! Local 2D Eikonal update stencil (1st order upwind scheme).

/// Solves the local 2D Eikonal equation to compute the candidate traveltime at a target node.
///
/// # Arguments
/// - `tx_min`: Smallest known traveltime along the X axis (left or right neighbor), if any.
/// - `tz_min`: Smallest known traveltime along the Z axis (top or bottom neighbor), if any.
/// - `dx`: Grid spacing along the X axis (> 0.0).
/// - `dz`: Grid spacing along the Z axis (> 0.0).
/// - `velocity`: Local velocity at the target node (> 0.0).
///
/// # Returns
/// The updated traveltime candidate.
#[inline]
pub fn solve_eikonal_2d(
    tx_min: Option<f64>,
    tz_min: Option<f64>,
    dx: f64,
    dz: f64,
    velocity: f64,
) -> f64 {
    let slowness = 1.0 / velocity;

    match (tx_min, tz_min) {
        (Some(tx), Some(tz)) => {
            // Try 2D update
            let idx2 = 1.0 / (dx * dx);
            let idz2 = 1.0 / (dz * dz);

            let a = idx2 + idz2;
            let b = -2.0 * (tx * idx2 + tz * idz2);
            let c = (tx * tx) * idx2 + (tz * tz) * idz2 - slowness * slowness;

            let discriminant = b * b - 4.0 * a * c;

            if discriminant >= 0.0 {
                let t_candidate = (-b + discriminant.sqrt()) / (2.0 * a);
                // Causality condition: upwind information
                if t_candidate > tx && t_candidate > tz {
                    return t_candidate;
                }
            }

            // Fallback to 1D update along the minimum direction
            (tx + dx * slowness).min(tz + dz * slowness)
        }
        (Some(tx), None) => tx + dx * slowness,
        (None, Some(tz)) => tz + dz * slowness,
        (None, None) => f64::INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stencil_1d_cases() {
        let dx = 10.0;
        let dz = 10.0;
        let v = 2000.0; // slowness = 0.0005 s/m => 10m takes 0.005 s

        // Only X neighbor known: T = 1.0 + 10.0 / 2000.0 = 1.005
        let t_x = solve_eikonal_2d(Some(1.0), None, dx, dz, v);
        assert!((t_x - 1.005).abs() < 1e-12);

        // Only Z neighbor known
        let t_z = solve_eikonal_2d(None, Some(2.0), dx, dz, v);
        assert!((t_z - 2.005).abs() < 1e-12);

        // No neighbor known
        let t_none = solve_eikonal_2d(None, None, dx, dz, v);
        assert!(t_none.is_infinite());
    }

    #[test]
    fn test_stencil_2d_diagonal_propagation() {
        let dx = 1.0;
        let dz = 1.0;
        let v = 1.0; // slowness = 1.0
        // Symmetrical wavefront coming from (0,0) at T=0 to (1,0) at T=1 and (0,1) at T=1.
        // Target (1,1): theoretical diagonal distance = sqrt(2) ≈ 1.41421356...
        let t_diag = solve_eikonal_2d(Some(1.0), Some(1.0), dx, dz, v);
        let expected = 1.0 + 1.0 / (2.0_f64).sqrt(); // T = 1 + sqrt(2)/2 ≈ 1.7071 (approx 1st-order upwind)
        assert!((t_diag - expected).abs() < 1e-10);
        assert!(t_diag > 1.0);
    }

    #[test]
    fn test_stencil_causality_fallback_to_1d() {
        let dx = 1.0;
        let dz = 1.0;
        let v = 1.0;
        // Large discrepancy: tx = 0.0, tz = 100.0. The 2D candidate will violate causality (t < tz).
        // It must fallback to 1D update from tx: T = 0.0 + 1.0 = 1.0
        let t = solve_eikonal_2d(Some(0.0), Some(100.0), dx, dz, v);
        assert!((t - 1.0).abs() < 1e-12);
    }
}
