# Code Review Ticket Pack

Date: 2026-03-08  
Repository: `mbagalman/Tesseract`

## Scope Reviewed
- `tesseract_visualizer.py`
- `README.md`
- `requirements.txt`

## Findings (Prioritized)

### TSC-001 - [P1] Projection singularity handling loses sign and can flip geometry incorrectly
- Status: Done (code + test scaffold)
- Type: Bug
- Location:
  - `tesseract_visualizer.py:84`
  - `tests/test_tesseract_visualizer.py:45`
- Evidence:
  - Near the projection plane (`viewer_distance - w ~= 0`), the denominator is replaced with `EPSILON` unconditionally:
    - `denom = np.where(np.abs(denom) < EPSILON, EPSILON, denom)`
  - This collapses both small positive and small negative denominators to a positive value, creating discontinuous orientation changes and incorrect perspective behavior around the camera plane.
- Resolution implemented:
  - Near-plane denominator clamping now preserves sign (`+EPSILON` for non-negative, `-EPSILON` for negative).
  - Added test `test_project_4d_to_3d_preserves_sign_near_projection_plane`.
- Validation:
  - Covered by automated tests (`pytest`).

### TSC-002 - [P1] No automated tests for core geometry and projection correctness
- Status: Done
- Type: Omission
- Location: repository-level (no `tests/` present)
- Evidence:
  - Core math functions (`generate_hypercube_vertices`, `generate_hypercube_edges`, `rotation_matrix_4d`, `project_4d_to_3d`) have no regression coverage.
- Risk:
  - Future changes can silently break edge-count invariants, rotation properties, or projection stability.
- Progress made:
  - Added `tests/test_tesseract_visualizer.py` with tests for:
    - Vertex and edge invariants.
    - Rotation matrix orthonormality and invalid-axis errors.
    - Projection near-plane sign behavior.
    - Input validation error paths for projection and plotting.
  - Refactored plotting imports to load lazily, so core math tests run even in minimal environments.
  - Added CI workflow at `.github/workflows/tests.yml` to run `pytest` on pushes/PRs.
- Validation:
  - `python3 -m pytest -q` passes locally (`16 passed`).

### TSC-003 - [P2] Missing input validation on public math APIs
- Status: Done (code + test scaffold)
- Type: Omission / robustness
- Location:
  - `tesseract_visualizer.py:38`
  - `tesseract_visualizer.py:69`
  - `tesseract_visualizer.py:84`
  - `tesseract_visualizer.py:105`
  - `tests/test_tesseract_visualizer.py:31`
- Evidence:
  - No guardrails for invalid `axis1/axis2` values (out of range or equal).
  - `viewer_distance` is accepted without finite/positive checks.
- Risk:
  - Invalid caller input produces hard-to-debug runtime failures or unstable render behavior.
- Resolution implemented:
  - Added `_validate_axis_index` and `_validate_finite_number` helpers.
  - Added validation for axis indices, equal-axis rotations, non-finite angles, bad point array shapes, and non-positive/non-finite `viewer_distance`.
  - Added tests for invalid rotation axes/inputs and invalid projection/plot arguments.
- Validation:
  - Covered by automated tests (`pytest`).

### TSC-004 - [P3] Environment reproducibility not fully locked
- Status: Done
- Type: Omission
- Location:
  - `requirements.lock.txt`
  - `requirements-dev.lock.txt`
  - `README.md`
- Evidence:
  - Dependencies are lower-bounded but not pinned to tested versions.
- Risk:
  - Different environments may render differently or break with upstream major updates.
- Resolution implemented:
  - Added `requirements.lock.txt` with pinned runtime dependencies.
  - Added `requirements-dev.lock.txt` with pinned runtime + dev dependencies.
  - Documented pinned install and lock-refresh workflow in `README.md`.
- Validation:
  - Lock files generated from clean virtual environments via `pip freeze`.

## Current Summary
- Done: `TSC-001`, `TSC-002`, `TSC-003`
- Done: `TSC-004`

## Notes
- Most recent local test run: `python3 -m pytest -q` -> `16 passed in 0.04s`.
