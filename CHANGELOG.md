
#### `CHANGELOG.md`
Format standard [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) :

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 2D grid representation (`Grid2D`) with depth-positive convention and C-order indexing.
- 2D local upwind Eikonal stencil (`solve_eikonal_2d`).
- Narrow band priority queue (`NarrowBand`) and node state tracking (`NodeState`) with lazy deletion.