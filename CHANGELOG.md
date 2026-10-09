
#### `CHANGELOG.md`
Format standard [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) :

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 2D grid representation (`Grid2D`) with depth-positive convention and C-order indexing (Block 1).
- 2D local upwind Eikonal stencil (`solve_eikonal_2d`) (Block 1).
- Narrow band priority queue (`NarrowBand`) and node state tracking (`NodeState`) with lazy deletion (Block 2).
- 2D Fast Marching Method solver (`FmmSolver2D`) with analytical source initialization and heterogeneous velocity support (Block 3).
- Integration test suite validating homogeneous radial waves and two-layer Snell refraction / head waves (Block 3).
- Python bindings via PyO3 and Maturin exposing `seikonal.solve_2d` with zero-copy NumPy interoperability (Block 4).
- Pytest test suite and Python CI pipeline integration (Block 4).