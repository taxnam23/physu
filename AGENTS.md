# AGENTS.md — `physu` Project Guidelines

## 1. Project Overview
- **Package Name (PyPI & `import`):** `physu` (`pip install physu` / `import physu`).
- **Purpose:** A Python library providing physics formulas, laws, and utilities, designed to scale progressively across independent physics domains/modules.
- **Core Differentiator:** Strict dimensional checking and physical unit handling using **`pint`** for all input and output quantities.

## 2. Language Policy
- **All project artifacts in English:** All source code, module names, function/class names, parameters, docstrings, comments, error messages, tests, README, and repository documentation (`AGENTS.md`, etc.) must be written in **English**.

## 3. Technical & Architectural Guidelines
- **Working Directory:** `d:\Proyectos\physicspythob`
- **Primary Dependency:** `pint` for units and dimensional validation.
- **Physical & Dimensional Validation:**
  - Validate dimensional compatibility for every physical parameter (e.g., length, time, mass).
  - Validate physical domain constraints (e.g., disallow negative mass or negative time where non-physical, negative discriminants under square roots, division by zero) by raising clear exceptions (`ValueError`, `pint.DimensionalityError`, or custom library exceptions).
- **Modular Scalability:**
  - Organize physics areas into independent submodules (e.g., `kinematics`, `dynamics`, `energy`, `thermodynamics`, `electromagnetism`, etc.) sharing a single centralized `pint.UnitRegistry` instance to prevent mismatches between `Quantity` objects from separate registries.
  - Keep naming, type annotations, and docstrings consistent across all modules.
- **Testing & Quality:**
  - Every new module or formula must include unit tests (e.g., with `pytest`) verifying: expected numerical results, conversions across compatible units, and expected failures on incompatible dimensions or physically invalid inputs.
