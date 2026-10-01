# Changelog

Every change to this repository is recorded here, in the same edit that makes it,
and the spec is brought in line with it (the rule is in `CLAUDE.md`). Newest first.
Each entry says what changed and why, and ends with a `Spec:` line: which sections
of the design spec and the decision register were updated to match, or that no
part of the spec describes the change.

## Unreleased

### 2026-10-01 - The design documents, under docs/design/

- Added the design spec (`docs/design/5d-chess-transformer-design.md`), the
  decision register (`docs/design/5d-chess-decisions.md`) and the general
  architecture the design instantiates
  (`docs/design/scalable-transformers-architecture.md`), as supplied: the current
  copies, carrying register row M11 and spec §15.6. They were kept untracked in a
  folder named `Specification documents/`; every reference to that path
  (`AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, the boundary's documentation list
  in `oracles/gate.toml`, and a harness test that classifies this repository's own
  files) now names `docs/design/`. Their text is unchanged: they are maintained
  upstream.
- These copies already carry the corrections the oracle work below found owing:
  register J20 removes the claims that cwmtt's checkmate tool uses z3 and that
  ftxi packs a coordinate into 8 bits, and H26 names the notation specification as
  the fourth submodule.
- Spec: these are the spec and the register; not otherwise changed.

### 2026-10-01 - GitHub remote; the graft shims carry no local path

- Added the GitHub repository as `origin`
  (https://github.com/Silicon-Spider/Temporal-5D). GitHub created it with a
  commit of its own - `LICENSE` (the GPL-3.0 text) and a two-line `README.md` -
  so the graft commit, never pushed, was rebased onto it: e1b6980 is now
  a56beb4. Nothing has been pushed yet. The project's licence is still being
  decided, and whether that `LICENSE` file stays is part of the decision. The
  oracle work stays uncommitted, as before. `AGENTS.md`, `CLAUDE.md`,
  `CONTRIBUTING.md` and `oracles/README.md` no longer say the repository has no
  remote: they say nothing has been pushed to it.
- `.claude/helpers/graft-hooks.cjs` and `.claude/helpers/graft-statusline.cjs`:
  the first place the shims look for graft (`BAKED`), which `graft init` writes
  as an absolute path into the installing user's profile, is now the same
  directory found from node's own install directory (npm's global root on
  Windows), so the commit carries no local path. Re-running `graft init` writes
  the absolute path again. `PROVENANCE.md` records the change.
- Spec: no section describes the developer tooling, the remote, or the
  project's own licence; not changed.

### 2026-10-01 - Every change goes in the changelog and the spec

- Added this file, and a rule - stated in full in `AGENTS.md` (for coding agents,
  and imported by `CLAUDE.md`) and `CONTRIBUTING.md` (for people), and pointed to
  from `CLAUDE.md` - that a change is finished only when it is recorded here and
  the design spec and the decision register say what is now true, in the same
  edit. The pin procedure in `CONTRIBUTING.md` gains the step that does this for
  a pin bump.
- Why: the register already requires a decision to change in the spec and the
  register together; this carries the same discipline to every change, and gives
  one place to see what changed.
- Spec: no section describes how changes are recorded; not changed. The entry
  below still owes its spec updates.

### 2026-09-30 to 2026-10-01 - The oracle side of the rules-engine gate (H27)

- Added `oracles/`, a separate, standard-library-only Python project: the
  differential-testing harness of spec §11.0 (decisions H23-H29). Commands:
  `sync`, `doctor`, `build`, `verify`, `selftest`, `gate`, `golden`, `observe`,
  `corpus`, `boundary`, `provenance`. Its manual is `oracles/README.md`.
- Pinned as git submodules under `oracles/third_party/`: ftxi/5dchess_engine
  (v0.4.10), penteract/cwmtt, 5d-chess-js (the rules library of Chess in 5D), and
  adri326/5dchess-notation as a source of game records. Commits, licences and
  toolchains are in `oracles/oracles.lock.toml`.
- Drivers of our own for ftxi (C++; part of it adapted from ftxi's tools, under
  its BSD-2-Clause notice) and 5d-chess-js (Node). cwmtt runs through its stock
  command-line program.
- A pinned corpus of 41 game records (37 read in place from the submodules, 4
  written here); each oracle's recorded answers (`oracles/golden/`); the generated
  `oracles/DISAGREEMENTS.md`; an empty `oracles/known-divergences.toml`.
- The boundary check that keeps the engine from using the harness or an oracle; a
  gate in three tiers (commit, nightly, release); two GitHub Actions workflows,
  which have never run (the repository has no remote).
- `PROVENANCE.md`, `CONTRIBUTING.md`, `AGENTS.md`, `CLAUDE.md`, `.gitattributes`,
  and `.gitignore` entries for the harness cache and Python byte-code.
- Why: H27 - the oracles and the gate exist before the rules engine, so that the
  gate cannot be skipped or bent to fit it.
- Spec: not yet brought in line. Pinning found statements in §11.0 and the register
  that are no longer true, listed in `PROVENANCE.md` under "Where the design spec
  and the pinned reality differ": cwmtt no longer uses z3; ftxi packs a coordinate
  into 32 bits, not 8 (one of H24's stated reasons); ftxi runs through a C++ driver
  and its stock CLI, not its Python bindings; a fourth repository, the notation
  specification, is pinned; and no oracle answers every question the gate asks.

### 2026-09-30 - graft wiring (commit e1b6980, rebased as a56beb4)

- Added the wiring of graft, a code-navigation tool for coding agents:
  `.claude/helpers/`, `.claude/skills/graft/`, and entries in
  `.claude/settings.json`, `.mcp.json`, `.ignore` and `.gitignore`. Developer
  tooling: nothing in the engine or the harness depends on it.
- Spec: not affected.
