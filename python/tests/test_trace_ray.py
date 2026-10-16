"""Integration tests for seikonal ray tracing Python bindings."""

import numpy as np
import pytest
import seiscode_eikonal as seikonal

def test_trace_ray_homogeneous_straight_line():
    """Verify that ray traced in homogeneous medium is a straight line connecting receiver to source."""
    nx, nz = 51, 51
    dx, dz = 10.0, 10.0
    v0 = 2000.0
    xs, zs = 100.0, 100.0
    xr, zr = 400.0, 300.0

    velocities = np.full((nx, nz), v0, dtype=np.float64)
    traveltimes = seikonal.solve_2d(velocities, dx, dz, xs, zs, order=2)

    ray = seikonal.trace_ray(traveltimes, dx, dz, xs, zs, xr, zr)

    assert isinstance(ray, np.ndarray)
    assert ray.ndim == 2
    assert ray.shape[1] == 2
    assert len(ray) > 2

    # First point is receiver, last point is source
    np.testing.assert_allclose(ray[0], [xr, zr], atol=1e-5)
    np.testing.assert_allclose(ray[-1], [xs, zs], atol=1e-3)

    # All points lie on straight line within half a cell tolerance
    for px, pz in ray:
        u = (px - xs) / (xr - xs)
        expected_z = zs + u * (zr - zs)
        assert abs(pz - expected_z) < 0.5 * dx


def test_trace_rays_multi_receivers():
    """Verify multi-receiver ray tracing."""
    nx, nz = 51, 51
    dx, dz = 10.0, 10.0
    v0 = 2000.0
    xs, zs = 250.0, 250.0

    velocities = np.full((nx, nz), v0, dtype=np.float64)
    traveltimes = seikonal.solve_2d(velocities, dx, dz, xs, zs, order=2)

    receivers = np.array([
        [100.0, 100.0],
        [400.0, 100.0],
        [400.0, 400.0],
    ], dtype=np.float64)

    rays = seikonal.trace_rays(traveltimes, dx, dz, xs, zs, receivers)

    assert len(rays) == 3
    for i, ray in enumerate(rays):
        assert isinstance(ray, np.ndarray)
        np.testing.assert_allclose(ray[0], receivers[i], atol=1e-5)
        np.testing.assert_allclose(ray[-1], [xs, zs], atol=1e-3)
