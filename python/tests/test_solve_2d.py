"""Integration tests for seikonal Python bindings."""

import numpy as np
import pytest
import seikonal


def test_solve_2d_homogeneous_analytical():
    """Verify that Python wrapper computes accurate traveltimes matching the analytical solution."""
    nx, nz = 51, 51
    dx, dz = 10.0, 10.0
    v0 = 2000.0  # 2000 m/s
    xs, zs = 250.0, 250.0

    velocities = np.full((nx, nz), v0, dtype=np.float64)

    traveltimes = seikonal.solve_2d(
        velocities=velocities,
        dx=dx,
        dz=dz,
        xs=xs,
        zs=zs,
        x0=0.0,
        z0=0.0,
    )

    assert isinstance(traveltimes, np.ndarray)
    assert traveltimes.shape == (nx, nz)
    assert traveltimes.dtype == np.float64

    # Compare against analytical solution
    x = np.arange(nx) * dx
    z = np.arange(nz) * dz
    X, Z = np.meshgrid(x, z, indexing="ij")
    distances = np.sqrt((X - xs) ** 2 + (Z - zs) ** 2)
    t_exact = distances / v0

    # Away from source (dist > 2 * dx), check relative error bounds
    mask = distances > 2.0 * dx
    rel_error = np.abs(traveltimes[mask] - t_exact[mask]) / t_exact[mask]

    assert np.mean(rel_error) < 0.05
    assert np.max(rel_error) < 0.10


def test_solve_2d_two_layer_head_wave():
    """Verify that head-wave arrives before direct wave in a two-layer model."""
    nx, nz = 101, 51
    dx, dz = 10.0, 10.0
    v1, v2 = 1500.0, 3000.0
    z_interface = 150.0

    velocities = np.empty((nx, nz), dtype=np.float64)
    velocities[:, :15] = v1  # z < 150 m
    velocities[:, 15:] = v2  # z >= 150 m

    traveltimes = seikonal.solve_2d(velocities, dx=dx, dz=dz, xs=0.0, zs=0.0)

    # Far offset on surface: x = 900 m, z = 0 m (index: ix=90, iz=0)
    t_calc = traveltimes[90, 0]
    t_direct = 900.0 / v1
    t_head_exact = (900.0 / v2) + 2.0 * z_interface * np.sqrt(v2**2 - v1**2) / (v1 * v2)

    assert t_calc < t_direct
    assert np.isclose(t_calc, t_head_exact, rtol=0.05)


def test_solve_2d_input_validation_errors():
    """Verify that invalid inputs raise Python ValueError / IndexError properly (no panic)."""
    valid_vel = np.full((10, 10), 1000.0, dtype=np.float64)

    # 1. Non-positive velocity
    bad_vel = np.full((10, 10), -500.0, dtype=np.float64)
    with pytest.raises(ValueError, match="Invalid velocity"):
        seikonal.solve_2d(bad_vel, dx=1.0, dz=1.0, xs=0.0, zs=0.0)

    # 2. Non-positive spacing
    with pytest.raises(ValueError, match="Invalid grid spacing"):
        seikonal.solve_2d(valid_vel, dx=-1.0, dz=1.0, xs=0.0, zs=0.0)

    # 3. Non C-contiguous array (Fortran-order)
    f_order_vel = np.asfortranarray(valid_vel)
    with pytest.raises(ValueError, match="must be C-contiguous"):
        seikonal.solve_2d(f_order_vel, dx=1.0, dz=1.0, xs=0.0, zs=0.0)