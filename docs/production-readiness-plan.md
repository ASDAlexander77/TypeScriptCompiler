# Production-readiness plan

Draft, 2026-11-13. Status: proposal, nothing here is started.

This plan takes tslang from "pre-alpha that passes its own tests" to a release a stranger can depend
on. It is ordered by phases with exit gates, not dates. Facts are marked **[checked]** (verified in
the repo or on GitHub on 2026-11-13) or **[notes]** (from earlier session notes, to be re-verified).

## 1. Decisions made in this draft

These four calls shape everything below. Each one is a proposal and can be overridden.

| Decision | Proposed | Why |
| --- | --- | --- |
| Positioning | A documented TypeScript subset with listed differences from `tsc`, not a drop-in replacement | Some differences are deliberate: integer literals are typed `s32`, user functions are global symbols that can clash with C library names, and `-mm=own` rejects programs `tsc` accepts **[notes]**. "Full compatibility" cannot be claimed. |
| First production tier (confirmed by the owner, 2026-11-13) | Windows x64 and Linux x64, ahead-of-time executables and shared libraries, `-mm=gc` | It is the default model and the oldest path. The README itself calls `-mm=own` "in development" **[checked]**. |
| Later tiers | Tier 2: JIT, `-mm=rc`, wasm32. Tier 3: `-mm=own`, Android, wasm64, 32-bit x86 | Each is newer or has a known structural limit (rc does not collect cycles). |
| Non-goals for 1.0 | macOS, a language server, npm package resolution, running untyped JavaScript | None is among the README's targets or features; each is a project of its own. |

"Production-ready" for a tier means all of the following hold:

1. **No silent miscompiles known.** Every known wrong-output bug is fixed or the construct is rejected
   with a diagnostic.
2. **No compiler crashes on valid or invalid input.** Unsupported constructs give an error with a
   source location.
3. **The supported subset is written down** and tested against outside code, not only this repo's
   tests.
4. **A release can be stayed on.** There is a stable channel, a version policy and a changelog.
5. **Regressions are caught before release** by an assertions-enabled compiler build in CI.

## 2. Where it stands

**Strengths [checked]**
- The issue tracker has zero open issues. One PR is open: #435, the LLVM 23 update.
- CI builds the compiler on Windows and Linux and runs the suite as release and debug *programs*,
  each as executable and JIT, plus wasm32/wasm64 under Node. A separate workflow runs Android tests
  on an x86_64 emulator (API 29).
- The tester corpus holds 865 `.ts` files. The last recorded full run was about 6,600 ctest tests
  green **[notes]**.
- Releases are built from a pushed `v*` tag and run the debug-program legs before publishing.
- `LICENSE`, `CODE_OF_CONDUCT.md` and issue templates exist.

**Gaps [checked]**
- **The compiler itself is only ever built as Release in CI.** Assertions and `llvm_unreachable` are
  compiled out, so a whole class of internal errors is invisible there.
- **42 `llvm_unreachable` / `report_fatal_error` sites** remain across 23 files under `tslang/`. Each
  is a potential crash instead of a diagnostic.
- **No systematic outside measurement.** The corpus appears to include some cases ported from
  TypeScript's own tests (for example `logicalAssignment5_3`, `typeGuardFunction` **[notes]**; not
  counted), but there is no run of the full official suite and no third-party code beyond
  `native_ts_graphics`.
- **No stable channel.** Every release is a pre-release (alpha127 today, five in four days).
- **Missing files:** `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md` (not in the root, `.github/`
  or `docs/`).
- **No limits documentation.** The docs describe features. There is no supported-subset page and no
  differences-from-`tsc` page. The roadmap in `docs/whats-new.md` is three completed items.
- **The scheduled test run is weekly** (Monday 18:30 UTC). The push-to-`main` trigger is commented
  out in both test workflows.

**Gaps [notes], to re-verify**
- The Debug-built compiler had 38 known suite failures on 2026-10-01, including a JIT `--opt` abort
  on any program. Not re-run since.
- Known open items that are not in the tracker:
  - `-mm=own`: optional accessor fields (capturing callbacks in Map/Set work since 2026-11-16)
  - memory-model fuzzer findings R1-R10, O4, G1
  - run-time `Symbol` values and `obj[sym]`
  - a `(() => T) | null` field cast
  - a closure stored in a variable and passed to a `@cdecl` parameter loses its captures (recorded in
    `TSLANG_ISSUES.md`) **[checked]**
  - flaky tests: `check_declarations` (union order by pointer) and some async tests

## 3. Phases

### Phase 0: make the unknowns visible

Nothing is fixed in this phase. It produces the lists the later phases work through.

| Step | Output |
| --- | --- |
| Re-run the full suite on a Debug-built compiler, Windows and Linux | Current failure list, replacing the 2026-10-01 number |
| File every known untracked item above as a GitHub issue with a label per tier | The tracker becomes the single backlog |
| Triage the 42 fatal-error sites: reachable, proven dead, or unknown | A table in the tracker; reachable ones become issues |
| Land or drop the uncommitted edits in `tslang/CMakeLists.txt` and `docs/how/wasm/test/test_wasm.ps1` | Clean `main` |

**Exit gate:** every known defect is an issue, and the Debug failure list is current.

### Phase 1: external measurement

The aim is to find what the project's own tests cannot.

1. **TypeScript conformance triage.** Count the cases already ported into the corpus, then run the
   full official TypeScript `tests/cases/conformance` and `compiler` cases through tslang. Classify each as: passes, clean diagnostic, compiler crash, or
   wrong output. Publish the totals per language area. The crash and wrong-output buckets are the
   work list. The clean-diagnostic bucket becomes the unsupported list in the docs.
2. **Differential runs against Node.** For every test that prints output and uses no tslang-only
   types, run the same file under Node and compare. Differences are either bugs or entries for the
   differences page.
3. **Real programs.** Port three to five small third-party TypeScript programs with no DOM or Node
   dependencies (a JSON parser, a markdown renderer, an algorithm library, a small interpreter).
   Record every change the port needed.
4. **Fuzzing beyond memory models.** Extend `tslang/test/fuzz` to generate type-level and
   control-flow programs, checked against Node output.
5. **Sanitizers.** Run the suite with an AddressSanitizer build of the compiler on Linux, and run
   compiled programs under ASan for the gc and rc models.

**Exit gate:** conformance totals are published, and every crash and wrong-output result has an issue.

### Phase 2: tier 1 hardening

Scope: Windows and Linux x64, AOT, `-mm=gc`.

1. **Assertions-enabled CI leg.** Add a Windows and a Linux job that builds the compiler with
   assertions on (RelWithDebInfo plus `LLVM_ENABLE_ASSERTIONS`, or Debug) and runs the suite. It
   starts as allowed-to-fail with the Phase 0 list and becomes required when the list is empty.
2. **Fix the Phase 0 Debug failures.**
3. **Zero reachable fatal errors.** Convert each reachable site from the Phase 0 triage to a
   diagnostic with a location.
4. **Fix all tier 1 crash and wrong-output results from Phase 1.**
5. **Diagnostics quality.** Every error has a file, line and column, and where TypeScript has a
   matching error code, the same code. Add a test corpus of expected diagnostics.
6. **Garbage-collector soak tests.** Long-running allocation, thread and async workloads (hours, not
   seconds) on both platforms, watching for growth and crashes.
7. **Remove flakiness.** Fix or quarantine `check_declarations` and the async tests so a red run
   always means a real failure.

**Exit gate:** the assertions-enabled leg is green and required; zero open tier 1 issues labelled crash
or miscompile; a 24-hour soak passes on both platforms.

### Phase 3: release engineering and docs

This can run alongside Phase 2.

1. **Version policy.** Adopt semantic versioning. State what a minor release may change: language
   behaviour, command-line flags, runtime ABI, default library API.
2. **Two channels.** Keep the alpha tag stream as "nightly". Add a stable branch that only receives
   fixes, with release candidates before each stable release.
3. **Release gate.** A stable release requires the full matrix green, including the
   assertions-enabled leg and the Debug default library check that today is done by hand **[notes]**.
4. **Compiler and runtime compatibility.** The compiler already checks the async runtime's age
   before linking a library. Extend this to one version stamp shared by the compiler, runtime and
   default library, checked at link time.
5. **LLVM policy.** Pin one LLVM version per stable line. Decide whether #435 (LLVM 23) lands before
   or after 1.0.
6. **Project files.** Add `CHANGELOG.md`, `CONTRIBUTING.md` and `SECURITY.md` with a private
   reporting address. Remove the stale tracked root files `.travis.yml` and
   `build_tslang_release.txt`.
7. **Documentation.**
   - `docs/supported-typescript.md`: the subset, by language area, generated from the Phase 1 triage
   - `docs/differences-from-tsc.md`: every deliberate difference with a short example
   - `docs/status.md`: the tier table, what each tier guarantees and known limitations
   - a first-run guide checked on a clean machine for Windows and Linux
   - the README quick start leads with `--emit=exe`, and marks the JIT as preview, since the JIT is
     outside tier 1
   - a real roadmap in place of the three checked items in `docs/whats-new.md`
8. **Installers.** A Windows zip and Linux tarball exist. Add checksums and, if wanted, signing and a
   package-manager entry (winget, apt or Homebrew for Linux).

**Exit gate:** a release candidate can be installed and used on a clean machine from the docs alone.

### Phase 4: 1.0 for tier 1

1. Publish release candidates and ask for outside testing through GitHub Discussions.
2. Hold a fixed period (for example four weeks) with no new crash or miscompile reports against the
   candidate.
3. Publish 1.0 with tier 1 marked stable and tiers 2 and 3 marked preview or experimental in
   `docs/status.md`.

### Phase 5: promote tiers 2 and 3

Each tier repeats the Phase 2 gate for its own scope before it is marked stable.

| Target | Extra work before promotion |
| --- | --- |
| JIT | Fix the Debug `--opt` abort; decide whether the JIT cache format is versioned |
| `-mm=rc` | Document cycles as a limit or add a cycle detector; close the fuzzer rc findings |
| wasm32 | Stable host interface for the JavaScript runtime; browser test run, not only Node |
| `-mm=own` | Close open items (optional accessor fields, capturing callbacks); a written specification of the ownership rules; error messages reviewed by someone new to the model |
| Android, wasm64, x86 | Emulator runs for every ABI on each release; device testing |

## 4. Order and dependencies

- Phase 0 comes first and is small.
- Phases 1 and 3 can start together after Phase 0.
- Phase 2 needs the Phase 0 lists and grows as Phase 1 reports arrive.
- Phase 4 needs Phases 2 and 3 complete.
- Phase 5 work may continue throughout, but it must not block the tier 1 release.

## 5. Risks

| Risk | Mitigation |
| --- | --- |
| The conformance triage shows a much larger unsupported area than expected | Narrow the documented subset; do not delay 1.0 to widen it |
| One maintainer | `CONTRIBUTING.md`, labelled starter issues, and written design notes for the memory models and exception handling |
| New feature work keeps resetting stability | Feature work lands on `main` only; the stable branch takes fixes only |
| LLVM upgrades change behaviour | One LLVM version per stable line; upgrade only at a minor release with the full matrix |

## 6. Open questions for the owner

1. Should the test workflows run on every push to `main`? The trigger is commented out today, which
   looks deliberate.
2. Is the "documented subset" positioning acceptable, or is closer `tsc` compatibility a goal?
3. Should LLVM 23 land before 1.0?
4. Is there an outside user or project, besides `native_ts_graphics`, whose needs should set
   priorities?
