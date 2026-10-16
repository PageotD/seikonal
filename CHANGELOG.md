# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-10-16

### Bug Fixes

- Format lib.rs
- *(ci)* Update cargo-deny config
- Manage warning and error/vulnerabilities
- *(ci)* Use pip install -e . for universal python dev installation
- Fix the build wheels for windows (#9)

### CI/CD

- Integrate python build and pytest tasks into Taskfile and GitHub Actions workflow
- Remove macos support (#10)
- Try to locate license file in sdist (#11)

### Documentation

- Aad features to changelog
- Update CHANGELOG

### Features

- Initial project structure
- Manage errors
- Add grid 2D
- Add 2D eikonal solver
- Define NodeState and min-heap OrderedNode for narrowband
- Implement NarrowBand container with state management and lazy deletion
- Implement 2D Fast Marching Method solver loop
- Implement PyO3 module and solve_2d Python wrapper
- *(stencil)* Implement 2nd-order upwind local eikonal formulation
- *(solver)* Integrate 2nd-order stencil support into 2D FMM solver
- Expose configurable 2nd-order stencil to Python and validate accuracy gains
- *(ray)* Implement node gradient and bilinear gradient interpolation
- *(ray)* Implement 2D ray tracing with Runge-Kutta 2 integration and straight-line validation
- *(python)* Expose trace_ray and trace_rays bindings with pytest suite

### Miscellaneous Tasks

- *(ci)* Update deny.toml to official template with allowed licenses

### Refactor

- Rename project, crates and package to seiscode-eikonal

### Style

- Fix formatting in solver integration tests

### Tests

- Add integration tests
- Validate 2D FMM solver on full homogeneous grid against analytical solution
- Add two-layer Snell refraction and head-wave validation test
- *(python)* Add pytest suite for solve_2d wrapper and input validation

