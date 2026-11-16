# AGENTS.md — Working Guidelines for `physu`

- **Language:** All code, identifiers, docstrings, comments, tests, changelogs, and docs must be in **English**.
- **Architecture:** Organize domains as subpackages under `src/physu/<domain>/` with focused submodules re-exported in `__init__.py` (see [README.md](README.md) for current modules and dev commands).
- **Pint & Validation:**
  - Never create a second `UnitRegistry`. Use `ureg`, `Quantity`, `QuantityLike`, `to_si`, and `to_unit` from `physu.units`.
  - Validate inputs with `validate_quantity` / `require_*` (`physu.validation`) and raise `PhysicalDomainError` (`physu.exceptions`) for non-physical inputs.
  - Reuse constants (`physu.constants`) and existing formulas instead of duplicating equations.
  - Pint gotchas: use `check()` (not `is_compatible_with("[...]")`); validate angles with `"rad"` / `"rad / s"`; use `to_unit` or `to_si` so `mypy --strict` passes cleanly.
- **Git & Ignored Files:**
  - Work on `main`. Branch/tag `0.1` are frozen for `0.1.x` hotfixes only (use explicit refspecs like `git push origin refs/heads/main`).
  - Never commit git-ignored local files (`ROADMAP.md`, `playground.py`, `.venv/`, `dist/`).
- **Done Criteria:** Run `pytest`, `ruff`, and `mypy` from `.venv\Scripts\` (separated with `;` in PowerShell) with zero errors, and update `README.md` plus `changelogs/<version>.md` for any new public feature.
