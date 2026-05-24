# Changes Authored by andreyDagger

Generated on 2026-05-24 for repository `andreyDagger/newclid-with-new-theorems`.

This report summarizes the repository changes authored by `andreyDagger` according to the local Git history and the additional drawer update prepared in this working copy. It intentionally describes the real commit history and does not rewrite or backdate commits.

## Commit History

### 2026-03-24 — `238a3472` — `initial commit`

Initial theorem/proof integration work across Newclid and Yuclid:

- Extended `newclid/src/newclid/all_rules.py`.
- Added Yuclid matcher/theorem implementation changes in `yuclid/src/matcher.cpp`, `yuclid/src/matcher.hpp`, `yuclid/src/theorem.cpp`, and `yuclid/src/theorem.hpp`.
- Updated the Python Yuclid adapter in `yuclid/python/py_yuclid/yuclid_adapter.py`.
- Added helper script `my_script.py`.
- Added generated proof/output artifacts under `output/` and `output.zip`.

Git stat: 12 files changed, 1126 insertions.

### 2026-05-19 — `ce4e7745` — `ncyclic`

Added support for the `ncyclic` predicate/statement path:

- Added Newclid predicate registration in `newclid/src/newclid/predicates/_index.py`.
- Added cyclic predicate logic in `newclid/src/newclid/predicates/cyclic.py`.
- Added Yuclid `ncyclic` statement files:
  - `yuclid/src/statement/ncyclic.cpp`
  - `yuclid/src/statement/ncyclic.hpp`
- Registered the new statement in `yuclid/src/CMakeLists.txt`.
- Updated matcher/theorem integration in `yuclid/src/matcher.cpp` and `yuclid/src/theorem.cpp`.
- Updated generated proof/output artifacts.

Git stat: 11 files changed, 635 insertions, 452 deletions.

### 2026-05-21 — `45711586` — `theorems and tests`

Expanded theorem and test integration:

- Extended theorem/rule registration in `newclid/src/newclid/all_rules.py`.
- Added test coverage in `my_tests.py`.
- Updated `my_script.py`.
- Updated Yuclid adapter, matcher, and theorem headers/implementations:
  - `yuclid/python/py_yuclid/yuclid_adapter.py`
  - `yuclid/src/matcher.cpp`
  - `yuclid/src/matcher.hpp`
  - `yuclid/src/theorem.cpp`
  - `yuclid/src/theorem.hpp`
- Added `my_code.patch` containing the patch snapshot.
- Updated generated proof/output artifacts.

Git stat: 12 files changed, 644 insertions, 686 deletions.

### 2026-05-24 — `ae8261fe` — `drawer`

Cleaned up a patch artifact:

- Removed `my_code.patch`.

Git stat: 1 file changed, 338 deletions.

### 2026-05-24 — `57a0d95d` — `draweer final`

Added the first version of the numerical geometry drawer:

- Added `drawer/geometry_drawer.py`, a PyTorch-based optimizer for formal geometry descriptions.
- Added example geometry descriptions under `drawer/examples/`.
- Added generated example diagrams under `drawer/out/`.
- Added unit tests in `drawer/tests/test_geometry_drawer.py`.

The initial drawer supported commands such as `cong`, `coll`, `cyclic`, `eqangle`, `midpoint`, `parallel`, `perp`, fixed coordinates, initial coordinates, and report-only checks.

Git stat: 11 files changed, 898 insertions.

## Drawer Update Prepared in This Working Copy

The current working copy extends the `drawer/` tool beyond the committed 2026-05-24 version.

Changed files:

- `drawer/geometry_drawer.py`
- `drawer/geometry_constraints.py`
- `drawer/tests/test_geometry_drawer.py`
- `drawer/README.md`

Main additions:

- Split differentiable geometry primitives and losses into `drawer/geometry_constraints.py`.
- Added new constraint commands:
  - `para a b c d`
  - `dists_equation v1 a1 b1 ... vn an bn target`
  - `angles_equation v1 a1 b1 c1 ... vn an bn cn target`
  - `eqratio a b c d e f g h`
  - `convex_polygon a1 a2 ... an`
- Added constructive point assignments:
  - `x = intersect a b c d`
  - `h = orthocenter a b c`
  - `i = incenter a b c`
  - `ia = excenter a b c`
- Constructive points are initialized from their formulas before optimization unless they are fixed.
- Added parallel optimization of independent restarts via `ThreadPoolExecutor`.
- Added early stopping when any worker reaches `--success-loss`.
- Added a `rich.Live` / `rich.Table` terminal progress view showing each worker's status, step, current loss, constraint loss, separation penalty, and best loss.
- Added CLI options:
  - `--threads`
  - `--success-loss`
  - `--no-live`
- Updated plotting so the new constraints and constructions draw relevant guide segments.
- Added README documentation for all new commands and parallel optimization.
- Expanded unit tests to cover the new constraints, constructive assignments, formula-based initialization, parser validation, and parallel restart stopping.

Verification:

```text
python -B -m unittest discover -s tests
Ran 30 tests in 83.502s
OK
```

## Notes for Review

- The commit history already contains authored work by `andreyDagger` from 2026-03-24 through 2026-05-24.
- No commit dates in this report are fabricated.
- The drawer update in this working copy is intended to be committed as a normal current-date commit.
