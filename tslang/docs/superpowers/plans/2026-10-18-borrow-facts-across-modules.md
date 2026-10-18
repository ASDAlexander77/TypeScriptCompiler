# Borrow Facts Across Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A library records which parameters of its non-private functions are only borrowed; a
program linked statically against it (or running under the JIT against the default library)
reads those facts, and stack promotion stops treating such calls as escapes.

**Architecture:** A new pass writes the facts of every module into one exported `[N x i8]`
constant, `__tsborrow_<file>_<hash>`, computed by `StackEscapeFacts` in a conservative mode. The
driver reads every `__tsborrow_` symbol from the archives an `--emit=exe` link uses (the default
library's static archive and `--lib` archives) or, under the JIT, from the default library DLL it
loads, and hands the merged text to the passes through `CompileOptions`. `StackEscapeFacts`
consults it for calls to bodiless functions.

**Tech Stack:** C++17, MLIR (TypeScript dialect, LLVM dialect), `llvm::object` (COFF, ELF,
archives), CMake/CTest scripts.

**Spec:** `tslang/docs/superpowers/specs/2026-10-18-borrow-facts-across-modules-design.md`

## Global Constraints

- Facts are generated, never hand-written. No attribute, no flag that asserts a borrow.
- A fact may only claim a parameter is borrowed when the library's own analysis proves it. When
  unsure: escape.
- Facts are read only for static links (`--emit=exe` with the default library's archive, `--lib`
  archives) and the JIT (the default library DLL it loads). Never for DLL links, never for
  `-shared` imports, never for `--emit=obj|llvm|bc|asm|mlir*` or `--emit=dll`.
- The file read must be the file linked or loaded.
- Missing, old or unreadable facts are not an error and print no warning: every call is an escape.
- Facts are emitted whether or not `--no-stack-promotion` is given, and whatever `--export=`,
  `--embed-declarations=` say.
- Producer: every class counts as exported (no vtable is trusted as local).
- Consumer: a virtual call on a library class stays an escape.
- Platforms: Windows, Linux, and the JIT.
- Version 1 may be almost pure plumbing; no default-library code changes.
- Full suite passes with promotion on and with `-DTSLANG_TEST_NO_STACK_PROMOTION=ON`.

## Review Focus

1. **A library function that keeps its argument in a way the analysis misses** (stored through
   an alias, captured by a closure that is returned, handed to a callee in another object file
   of the same library). Expected: not listed. Task 1 tests keep/return/callback/other-module
   cases.
2. **A `--lib` value the linker resolves to a different file than the one read.** Expected: the
   driver reads only a file it then passes to the linker by full path (Task 3 ruling).
3. **Symbol name mismatch between the library's definition and the program's declaration**
   (methods, namespaces). Expected: no match means no fact, never a wrong fact. Task 3 tests a
   namespaced function.
4. **A stale JIT cache after the default library is rebuilt with different facts.** Expected: a
   cache miss. Task 4 test.
5. **Archives with members that are not objects of the target format** (import members in a
   Windows import library, LLVM bitcode members). Expected: skipped silently. Task 2 test with an
   import library.

---

### Task 1: Producer — `__tsborrow_` symbol

**Files:**
- Create: `lib/TypeScript/BorrowFactsExportPass.cpp`
- Modify: `lib/TypeScript/StackEscapeFacts.h` (conservative mode, public query)
- Modify: `include/TypeScript/Passes.h`, `include/TypeScript/Defines.h`,
  `lib/TypeScript/CMakeLists.txt`, `tslang/transform.cpp`
- Test: `test/tester/stack-promotion/facts_lib.ts`, `test/tester/stack-promotion/facts_lib_other.ts`,
  `test/tester/CMakeLists.txt` (stack-promotion section)

**Interfaces:**
- Produces: `#define SHARED_LIB_BORROW_FACTS "__tsborrow_"` in Defines.h.
- Produces: `StackEscapeFacts(mlir::ModuleOp module, bool trustLocalVTables = true,
  const BorrowFactsMap *imported = nullptr)` (the third parameter is added in Task 3; add it now
  defaulted to null, unused) and `bool paramBorrowed(mlir_ts::FuncOp funcOp, unsigned index) const`
  (true only for a function with a body whose parameter never escapes).
- Produces: `std::unique_ptr<mlir::Pass> mlir_ts::createBorrowFactsExportPass(CompileOptions &)`.
- Produces: hidden option `--no-borrow-facts-export` (tslang.cpp, read in transform.cpp): skip
  the pass. Test-only stand-in for a library built before this change.
- Produces: facts text format, one line per function with at least one borrowed parameter:
  `<symbol> <index> [<index>...]\n`, symbol names sorted, indexes ascending, body-argument
  indexes (`this` is 0 for a method). The global holds the text plus a terminating `\0`.

- [ ] **Step 1: Write the failing test.** `facts_lib.ts` exports:
  `sumOf(a: number[]): number` (reads only → borrowed),
  `keep(a: number[])` (stores into a module global → not listed),
  `same(a: number[]): number[]` (returns it → not listed),
  `viaCallback(a: number[], f: (x: number[]) => void)` (calls `f(a)` → `a` not listed),
  `mixed(a: number[], b: number[]): number` (keeps `b`, reads `a` → `mixed 0`),
  `useOther(a: number[])` passing `a` to `otherKeep` declared from `facts_lib_other.ts` via
  `declare function` (no body here → not listed),
  `namespace Geo { export function len(a: number[]): number }` (borrowed, namespaced symbol),
  a private (non-exported, `--export` default) helper that borrows (must not appear), and a class
  `Holder` with a method `peek(a: number[])` that only reads `a` (listed as `<method symbol> 1`)
  plus a virtual method called through `this` inside the library to confirm vtables are not
  trusted. Register decision tests per model (gc rc own none):
  `tslang --emit=llvm --no-default-lib -mm=<mm> facts_lib.ts` with `PASS_REGULAR_EXPRESSION`
  `@__tsborrow_facts_lib_[0-9]+ = (dllexport )?constant \[[0-9]+ x i8\] c"[^"]*sumOf 0\\0A`
  and `FAIL_REGULAR_EXPRESSION` matching `keep |same |viaCallback |useOther |helper ` inside the
  constant. Add one more with `--no-stack-promotion --export=none` expecting the same symbol.
- [ ] **Step 2: Run** `ctest -C Release -R stack-facts-lib` — Expected: FAIL (no `__tsborrow_`).
- [ ] **Step 3: Implement.** In `StackEscapeFacts`, `trustLocalVTables=false` makes
  `collectVTables` treat every vtable as non-local. Add `paramBorrowed`. The pass builds
  `StackEscapeFacts(module, /*trustLocalVTables=*/false)`, walks non-declaration `FuncOp`s whose
  symbol visibility is not private, writes the text, and when it is non-empty creates an
  `LLVM::GlobalOp` (constant, external linkage, type `!llvm.array<N x i8>`, `value` the text with
  `\0`, section `DLL_EXPORT` so `ExportFixPass` makes it `dllexport` on Windows and clears the
  section). Name: `SHARED_LIB_BORROW_FACTS + stem + "_" + to_string(xxh3_64bits(canonical
  main-file path))`, the same suffix `MLIRGenImpl::moduleSymbolSuffix` builds (move that helper
  to a header both can use, or duplicate it beside a comment naming the original). Register the
  pass in `transform.cpp` right before the stack promotion block, unconditionally except
  `compileOptions.isJit`.
- [ ] **Step 4: Run** the step 2 command — Expected: PASS, all models.
- [ ] **Step 5: Run the stack-promotion section** `ctest -C Release -R stack -j8` — Expected: all
  pass (the new pass changes no promotion decision).
- [ ] **Step 6: Commit** `Borrow facts: libraries record borrowed parameters in __tsborrow_`.

### Task 2: Reader — `Dump::readCStringSymbols` and `--print-borrow-facts`

**Files:**
- Modify: `include/TypeScript/ObjDumper.h`, `lib/TypeScript/ObjDumper.cpp`
- Modify: `tslang/tslang.cpp` (hidden flag)
- Create: `test/tester/borrow-facts.cmake` (grows in Task 3)
- Test: `test/tester/CMakeLists.txt`

**Interfaces:**
- Consumes: the facts format and symbol from Task 1.
- Produces: `std::vector<std::pair<std::string, std::string>> Dump::readCStringSymbols(
  llvm::StringRef path, llvm::StringRef prefix)` returning (symbol, text) for every defined
  symbol with that prefix in an object file (COFF or ELF) or in every object member of an
  archive (`.lib`, `.a`). Members that are not COFF/ELF objects (import members, bitcode) and a
  file that cannot be opened yield nothing. The text is read from the symbol's section contents
  at the symbol's offset up to the first `\0` (bounded by the section end).
- Produces: hidden option `--print-borrow-facts=<file>`: prints each `symbol\ntext` found in
  `<file>` (object or archive) and exits 0, printing nothing for none.

- [ ] **Step 1: Write the failing test.** `borrow-facts.cmake`, invoked with `TSLANG`,
  `SOURCE_DIR`, `WORK_DIR`, `AR` (`${CMAKE_AR}`), `MM`: compiles `facts_lib.ts` with
  `--emit=obj --no-default-lib -mm=${MM}`, archives it (`lib /OUT:facts_lib.lib facts_lib.obj`
  on Windows, `ar rcs libfacts_lib.a facts_lib.o` elsewhere), then runs
  `--print-borrow-facts=` on the object and on the archive; both must match `sumOf 0` and
  `mixed 0`. On Windows also run it on `gc.lib`'s import library or any import `.lib` passed as
  `IMPORT_LIB` (Review Focus 5): exit 0, no output. Register `test-borrow-facts-<mm>` for the
  four models, WIN32 and UNIX.
- [ ] **Step 2: Run** `ctest -C Release -R test-borrow-facts` — Expected: FAIL (unknown option).
- [ ] **Step 3: Implement** `readCStringSymbols` with `llvm::object::createBinary`: an `Archive`
  iterates `children()` and recurses into each child that is an `ObjectFile`; an `ObjectFile`
  walks `symbols()`, keeps defined ones whose name starts with `prefix`, and reads
  `section->getContents()` at `symbol.getValue()` (COFF: the value is the section offset; ELF
  relocatable: likewise). Then the flag.
- [ ] **Step 4: Run** the step 2 command — Expected: PASS.
- [ ] **Step 5: Commit** `Borrow facts: read __tsborrow_ from objects and archives`.

### Task 3: Consumer — static links (`--lib` and the default library archive)

**Files:**
- Modify: `include/TypeScript/DataStructs.h` (`CompileOptions`)
- Modify: `lib/TypeScript/StackEscapeFacts.h`, `lib/TypeScript/StackPromotionPass.cpp`
- Create: `tslang/borrowfacts.cpp`
- Modify: `tslang/defaultlib.cpp`, `tslang/exe.cpp`, `tslang/tslang.cpp`, `tslang/CMakeLists.txt`
- Test: `test/tester/stack-promotion/facts_use.ts`, `test/tester/borrow-facts.cmake`,
  `test/tester/CMakeLists.txt`

**Interfaces:**
- Consumes: `Dump::readCStringSymbols` (Task 2); facts format (Task 1).
- Produces: `CompileOptions::importedBorrowFacts` (`std::string`, merged text, empty = none).
- Produces: `using BorrowFactsMap = llvm::StringMap<llvm::SmallVector<unsigned>>;` and
  `BorrowFactsMap parseBorrowFacts(llvm::StringRef text)` in `StackEscapeFacts.h` (malformed
  lines skipped).
- Produces: `std::string getDefaultLibBinary(bool shared, const CompileOptions &,
  const llvm::Triple &)` in `defaultlib.cpp`: the full path of the default library file a link
  or the JIT uses (`<dir>/TypeScriptDefaultLib.lib`, `.dll`, `libTypeScriptDefaultLib.a`, `.so`),
  composed with `getDefaultLibSubDir`; `exe.cpp` and `jit.cpp` derive their directory from it.
- Produces: `std::string resolveLinkLibrary(llvm::StringRef value)` in `borrowfacts.cpp`: the
  value itself when it names an existing file, else `<value>.lib` (Windows) or `lib<value>.a`
  (elsewhere) in the current directory when that exists, else empty.
- Produces: `void readStaticBorrowFacts(CompileOptions &)` in `borrowfacts.cpp`, called in
  `main` after `prepareDefaultLib` when `emitAction == BuildExe`.

- [ ] **Step 1: Write the failing test.** `facts_use.ts` declares the library's functions with
  `declare function` (and `declare namespace Geo`, and the `Holder` class with a factory
  `declare function makeHolder(): Holder`) and calls each with a fresh array literal of numbers
  that is otherwise only read. Extend `borrow-facts.cmake`: build the archive as in Task 2, then
  `tslang --emit=exe --no-default-lib -mm=${MM} --stack-promotion-report --lib=facts_lib
  facts_use.ts` in the work directory, and check the remarks:
  `sumOf`, `mixed`'s first argument, `Geo.len` → `stack`;
  `keep`, `same`, `viaCallback`, `mixed`'s second argument, `useOther` → `heap: unknown-callee`;
  `holder.peek(...)` (virtual) → `heap`.
  Run the exe; its printed result must equal the same program built with `--no-stack-promotion`.
  Also check: the same compile with `--emit=obj` reports every call `heap` (no facts read). And
  the missing-facts case: an archive of `facts_lib.ts` built with the hidden Task 1 flag
  `--no-borrow-facts-export` (it only suppresses the symbol, the safe direction) links cleanly,
  prints no warning, and reports every call `heap: unknown-callee`.
  Add the `-shared` negative: `test-runner -shared` style DLL of `facts_lib.ts` imported by
  `facts_import.ts` (an `import` version of `facts_use.ts`) with `--stack-promotion-report`:
  every call `heap`.
- [ ] **Step 2: Run** `ctest -C Release -R test-borrow-facts` — Expected: FAIL (`sumOf` is
  `heap: unknown-callee`).
- [ ] **Step 3: Implement.** `readStaticBorrowFacts` reads `getDefaultLibBinary(false, ...)`
  unless `noDefaultLib`, and every `--lib` value through `resolveLinkLibrary`, concatenating the
  texts into `importedBorrowFacts`. `exe.cpp`: a `--lib` value that `resolveLinkLibrary`
  resolves is passed to the linker as that full path instead of `-l<value>` (so the file read
  is the file linked); others keep `-l`. `StackPromotionPass` parses the text once and passes
  the map to `StackEscapeFacts`. In `argumentEscapes`, a candidate `FuncOp` with an empty body
  whose name is in the map and lists `argIndex` returns `""`; otherwise unchanged. Virtual
  candidates are never declarations of library methods in a way that reaches this path because
  `virtualCandidates` already fails on non-local vtables — keep it that way.
- [ ] **Step 4: Run** the step 2 command — Expected: PASS, all models.
- [ ] **Step 5: Teeth.** Temporarily make the producer list every parameter; rerun; Expected:
  the `keep` program's output differs or it crashes (the kept array is read after the frame is
  gone — print it from the global after the calling function returns). Revert.
- [ ] **Step 6: Run the whole suite** twice (promotion on; `-DTSLANG_TEST_NO_STACK_PROMOTION=ON`),
  `ctest -C Release -j16 --timeout 300 > <workspace>/suite.txt` — Expected: no new failures.
- [ ] **Step 7: Commit** `Borrow facts: static links read facts from --lib and the default library`.

### Task 4: JIT — the default library DLL, and the cache key

**Files:**
- Modify: `tslang/borrowfacts.cpp`, `tslang/jit.cpp`, `tslang/jitcache.cpp`, `tslang/tslang.cpp`
- Test: `test/tester/borrow-facts-defaultlib.cmake`, `test/tester/CMakeLists.txt`

**Interfaces:**
- Consumes: `getDefaultLibBinary(true, ...)` (Task 3), `Dump::getSymbols`.
- Produces: `void readJitBorrowFacts(CompileOptions &)`: when `isJit` and not `noDefaultLib`,
  loads the DLL with `llvm::sys::DynamicLibrary::getPermanentLibrary` (the same file
  `prepareJitProcess` loads), enumerates its `__tsborrow_` symbols with `Dump::getSymbols`
  (on ELF include the dynamic symbol table), and reads each as `(const char *)address`.
  Called before both `runJitCached` and the non-cached compile.
- Produces: the JIT cache manifest lists the default library DLL as a dependency hashed with
  `fileHash`, as `sharedLibraryImports` already are, so a rebuilt library (new facts) misses the
  cache. (Ruling: a file-hash dependency instead of a facts hash in `optionsKey`: it covers the
  facts and anything else in the DLL, and needs no extra key text.)

- [ ] **Step 1: Write the failing test.** `borrow-facts-defaultlib.cmake` (SKIPPED, like
  `defaultlib-collector.cmake`, when no default library is built for the model): a program
  using `s.replace(new RegExp("a", "g"), "b")` and a function from the default library that
  the producer lists; pick one from `--print-borrow-facts` on the built archive whose parameter
  is an object or array. If the library lists none, the test asserts only that facts were read:
  the archive's facts text is non-empty and the AOT and JIT compiles succeed with correct
  output. Check `--emit=exe` and `--emit=jit` remarks agree. Cache: run `--emit=jit` with the
  JIT cache on into a fresh cache folder, then check the manifest written there names the
  default library DLL among its dependencies.
- [ ] **Step 2: Run** `ctest -C Release -R borrow-facts-defaultlib` — Expected: FAIL or SKIPPED;
  if SKIPPED, build the default library for one model first (DefaultLib `build.bat`) so the
  test runs.
- [ ] **Step 3: Implement** as in Interfaces.
- [ ] **Step 4: Run** the step 2 command — Expected: PASS.
- [ ] **Step 5: Commit** `Borrow facts: the JIT reads the default library's facts; cache key`.

### Task 5: Documentation

**Files:**
- Modify: `docs/memory-models.md` (Stack promotion section)
- Modify: `tslang/docs/superpowers/specs/2026-10-18-borrow-facts-across-modules-design.md`
  (status line: implemented, PR number)

- [ ] **Step 1:** Replace "passed to a function in another module, including the default
  library;" with: passed to a function in another module, unless that module is linked
  statically (or is the default library under the JIT) and its own analysis showed the function
  only reads the argument; never through a DLL or a `-shared` import. Mention
  `--print-borrow-facts=<file>`.
- [ ] **Step 2: Commit** `Docs: borrow facts across modules`.
