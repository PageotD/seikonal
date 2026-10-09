# seikonal

An efficient, robust Fast Marching Method (FMM) eikonal solver written in pure Rust with Python bindings.

## Overview

`seikonal` solves the eikonal equation on 3D Cartesian grids (and spherical grids in future releases) for seismic traveltime computation.

- **`seikonal-core`**: High-performance, pure Rust numerical core without Python dependencies.
- **`seikonal-py`**: Lightweight, zero-copy Python bindings powered by PyO3 and Maturin.

## Development

Prerequisites:
- Rust >= 1.85 (Edition 2024)
- Python >= 3.11
- [Task](https://taskfile.dev)

Common tasks:
```bash
task lint       # Run format check and clippy
task test       # Run test suite
task audit      # Run cargo-deny and cargo-audit
task ci         # Run all local CI checks
```

## Note

AI were used to help me in the development of the tests and the review of the code.