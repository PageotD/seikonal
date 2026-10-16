//! Local 2D Eikonal update stencil (1st and 2nd order upwind schemes).

/// Computes the effective traveltime and grid spacing along an axis for upwind stencil.
///
/// If a second upwind neighbor is available and causal (`t1 > t2`), applies the 2nd-order
/// finite difference approximation:
/// `(3*T - 4*t1 + t2) / (2*h) = (T - (4*t1 - t2)/3) / (2/3 * h)`.
///
/// Otherwise, falls back to 1st-order: `(T - t1) / h`.
#[inline]
pub(crate) fn effective_axis_params(
    t1: Option<f64>,
    t2: Option<f64>,
    h: f64,
    order: usize,
) -> (Option<f64>, f64) {
    match (t1, t2) {
        (Some(t1_val), Some(t2_val)) if order >= 2 && t1_val > t2_val => {
            let t_eff = (4.0 * t1_val - t2_val) / 3.0;
            let h_eff = (2.0 / 3.0) * h;
            (Some(t_eff), h_eff)
        }
        (Some(t1_val), _) => (Some(t1_val), h),
        _ => (None, h),
    }
}

/// Solves the local 2D Eikonal equation with support for 1st or 2nd order upwind approximation.
///
/// # Arguments
/// - `tx1`: Smallest traveltime at 1st neighbor along X axis (ix ± 1).
/// - `tx2`: Traveltime at 2nd neighbor along the same direction in X (ix ± 2), if available and Alive.
/// - `tz1`: Smallest traveltime at 1st neighbor along Z axis (iz ± 1).
/// - `tz2`: Traveltime at 2nd neighbor along the same direction in Z (iz ± 2), if available and Alive.
/// - `dx`: Grid spacing along X (> 0.0).
/// - `dz`: Grid spacing along Z (> 0.0).
/// - `velocity`: Local velocity at target node (> 0.0).
/// - `order`: Stencil order (1 or 2).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn solve_eikonal_2d_order(
    tx1: Option<f64>,
    tx2: Option<f64>,
    tz1: Option<f64>,
    tz2: Option<f64>,
    dx: f64,
    dz: f64,
    velocity: f64,
    order: usize,
) -> f64 {
    let (tx_eff, dx_eff) = effective_axis_params(tx1, tx2, dx, order);
    let (tz_eff, dz_eff) = effective_axis_params(tz1, tz2, dz, order);
    let slowness = 1.0 / velocity;

    match (tx_eff, tz_eff) {
        (Some(tx), Some(tz)) => {
            let idx2 = 1.0 / (dx_eff * dx_eff);
            let idz2 = 1.0 / (dz_eff * dz_eff);

            let a = idx2 + idz2;
            let b = -2.0 * (tx * idx2 + tz * idz2);
            let c = (tx * tx) * idx2 + (tz * tz) * idz2 - slowness * slowness;

            let discriminant = b * b - 4.0 * a * c;

            if discriminant >= 0.0 {
                let t_cand = (-b + discriminant.sqrt()) / (2.0 * a);
                // Causality condition: candidate must be greater than 1st-order upwind inputs
                let tx_check = tx1.unwrap_or(tx);
                let tz_check = tz1.unwrap_or(tz);
                if t_cand > tx_check && t_cand > tz_check {
                    return t_cand;
                }
            }

            // Fallback to 1D update
            let t1d_x = tx1.map(|t| t + dx * slowness).unwrap_or(f64::INFINITY);
            let t1d_z = tz1.map(|t| t + dz * slowness).unwrap_or(f64::INFINITY);
            t1d_x.min(t1d_z)
        }
        (Some(_), None) => tx1.map(|t| t + dx * slowness).unwrap_or(f64::INFINITY),
        (None, Some(_)) => tz1.map(|t| t + dz * slowness).unwrap_or(f64::INFINITY),
        (None, None) => f64::INFINITY,
    }
}

/// Backward compatible 1st-order solver.
#[inline]
pub fn solve_eikonal_2d(
    tx_min: Option<f64>,
    tz_min: Option<f64>,
    dx: f64,
    dz: f64,
    velocity: f64,
) -> f64 {
    solve_eikonal_2d_order(tx_min, None, tz_min, None, dx, dz, velocity, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stencil_1d_cases() {
        let dx = 10.0;
        let dz = 10.0;
        let v = 2000.0;

        let t_x = solve_eikonal_2d(Some(1.0), None, dx, dz, v);
        assert!((t_x - 1.005).abs() < 1e-12);

        let t_none = solve_eikonal_2d(None, None, dx, dz, v);
        assert!(t_none.is_infinite());
    }

    #[test]
    fn test_stencil_order2_circular_wavefront_accuracy() {
        let dx = 1.0;
        let dz = 1.0;
        let v = 1000.0;

        // Circular wavefront from source (0, 0): T(x, z) = sqrt(x^2 + z^2) / v
        let t_exact_fn = |x: f64, z: f64| (x * x + z * z).sqrt() / v;

        // Target at (10, 10)
        let t_exact = t_exact_fn(10.0, 10.0);

        // 1st and 2nd neighbors in X: (9, 10) and (8, 10)
        let tx1 = t_exact_fn(9.0, 10.0);
        let tx2 = t_exact_fn(8.0, 10.0);

        // 1st and 2nd neighbors in Z: (10, 9) and (10, 8)
        let tz1 = t_exact_fn(10.0, 9.0);
        let tz2 = t_exact_fn(10.0, 8.0);

        // Order 1 estimate
        let t_ord1 = solve_eikonal_2d_order(Some(tx1), None, Some(tz1), None, dx, dz, v, 1);
        let err_ord1 = (t_ord1 - t_exact).abs() / t_exact;

        // Order 2 estimate
        let t_ord2 =
            solve_eikonal_2d_order(Some(tx1), Some(tx2), Some(tz1), Some(tz2), dx, dz, v, 2);
        let err_ord2 = (t_ord2 - t_exact).abs() / t_exact;

        // Order 2 must be significantly more accurate than Order 1 on curved circular wavefront
        assert!(
            err_ord2 < err_ord1,
            "Order 2 error ({err_ord2}) should be lower than Order 1 error ({err_ord1})"
        );
        assert!(err_ord2 < 0.005, "Order 2 error too high: {err_ord2}");
    }
}
