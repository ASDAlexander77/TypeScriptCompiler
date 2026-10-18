# Borrow facts across modules

Status: design, 2026-10-18. Not implemented. Follows
`2026-10-18-stack-promotion-design.md` (sections 8.1 and 8.2) and its v1 and v1.1 implementations
(PRs #605, #606).

## 1. Intent

Stack promotion treats any call into another module as an escape: the analysis has no body to
look at, so it cannot know the callee does not keep its argument. This adds the plumbing for one
module to tell another which parameters of its exported functions are only borrowed. The default
library is the first library that publishes these facts and the first one programs read them from.

The user's stated purpose is the plumbing itself. The gain from the default library alone is
expected to be modest: its non-generic API mostly takes numbers and strings (§4.2). The same
mechanism is what caller-provided result slots (project "a", next) and `-shared` modules (later)
will build on.

## 2. Decisions (the user's)

- **Scope of version 1: calls from a program into the default library.**
- **Facts are generated, never hand-written.** When a library is built, the compiler records, for
  every exported function, which parameters its body only borrows. It derives them from the same
  escape analysis stack promotion runs. There is no `@borrows` attribute and no hand-written claim:
  a wrong claim is a dangling stack pointer.
- **Facts are stored inside the library binary** as exported symbols, beside the existing
  `__tsmm_` and `__tsown_` markers, so they can never disagree with the code they describe.
- **The compiler reads the default library binary on every compile**, AOT and JIT alike.

## 3. Goals, non-goals, anti-goals

Goals
- A program compiled against a default library built with this change promotes objects passed to
  default-library functions whose parameter is only borrowed.
- The facts survive any rebuild of the library: they are regenerated with it.

Non-goals
- Hand-written facts, for C functions or anything else.
- Facts about results (a result that borrows an argument, or a fresh result): that is project (a).
- Changing what the default library's API looks like.

Anti-goals (failure even if it works)
- Any fact that claims a parameter is borrowed when the body may keep it.
- Facts read from a binary other than the one the program is linked against.
- A compile that fails, or warns, because the facts are missing or unreadable. Missing facts mean
  "escape", exactly as today.

## 4. Feasibility notes

### 4.1 What exists

- `StackEscapeFacts` (`tslang/lib/TypeScript/StackEscapeFacts.h`) already computes, for every
  function in a module, whether each parameter may escape (`paramEscapes`, a module fixpoint). An
  unknown callee is an escape; a virtual call is trusted only when every vtable holding that slot
  is local to the module.
- A library already exports text facts for importers: own's `__tsown_<file>_<hash>` symbol (a C
  string, one name per line), emitted by `OwnershipSignaturePass` (`SHARED_LIB_OWN_FACTS`,
  `Defines.h`). It is read by `MLIRGenModule.cpp` when importing a shared library: the symbol list
  comes from `Dump::getSymbols`, the text from `Dump::readExportedCString` (or from the loaded
  library under the JIT), and `addImportedOwnNoDrops` merges it into a module attribute.
- The default library reaches a program as hand-written declaration files (`lib.d.ts` and the files
  it references, in TypeScriptCompilerDefaultLib/src), and as a per-model binary under
  `{lib,dll}/<arch>/<vendor>/<os>/<env>/<mode>/<mm>` (`tslang --print-default-lib-dir`). Generic
  code (`lib.generics.ts`) is compiled into the program with its bodies, so it needs no facts.

### 4.2 What the default library's API can actually gain

Most non-generic default-library functions take numbers and strings, and strings are not
promotion candidates. The arguments that could move to the stack are `RegExp` objects
(`s.replace(new RegExp(...), x)`), rest-parameter arrays built at the call site
(`concat(...other: string[])`; promotable under gc/none only, since the elements are strings), and
small option objects (`toLocaleString(locale, { timeZone })`).

## 5. Design

### 5.1 Producing the facts (library side)

When a module is built as a library (`--emit=dll`, `--emit=lib`, and the object files the default
library build links), the compiler emits one exported symbol per module, in the style of
`__tsown_`. Suggested name: `__tsborrow_<file>_<hash>`, using the same suffix the other markers use.
Its value is a C string with one line per exported function that has at least one borrowed
parameter: the function's symbol name, then the indexes of its borrowed parameters (`this` is
parameter 0 of a method). The exact text format is the builder's call.

- The facts come from `StackEscapeFacts` run over the library module, at the same point stack
  promotion runs. They are emitted whether or not `--no-stack-promotion` is given: the flag decides
  what this module promotes, not what it promises importers.
- **Not tied to the `__tsmm_` marker (my call).** The default library's two builds emit no marker
  today: the DLL uses `--embed-declarations=false`, and the archive's object uses `--export=none`
  (DefaultLib `scripts/build_core.bat`). So the facts symbol is emitted on its own, for every
  function that is not private, whatever the export settings.
- **In a library build every class counts as exported (my call).** An importer can subclass any of
  them, even under `--export=none`, so the library's own analysis never trusts a virtual call
  through one of its classes' vtables. Otherwise a program subclass passed into the library could
  break a fact.
- Only functions that are not private get facts. A parameter is listed only when `paramEscapes` says it does
  not escape. Everything unlisted is an escape to the reader.
- A method of an exported class is listed under its own symbol. Whether a virtual call to it may be
  trusted is the reader's question (§5.3).

### 5.2 Reading the facts (program side)

When a program is compiled with the default library, the compiler opens the binary it will link:
the DLL under the JIT, and the static library or DLL it links ahead of time. It reads every
`__tsborrow_` symbol and records the facts on the module, as `addImportedOwnNoDrops` does for own,
for `StackEscapeFacts` to read.

- **Same binary.** The facts must come from the exact file the link uses. If the AOT link takes a
  static archive, the archive is read: `Dump` gains whatever archive support that needs. Reading
  the DLL beside it instead is not allowed, since the two could come from different builds.
- **Missing or unreadable facts are not an error.** No symbol, an old library, a file that cannot
  be read: no facts, every call an escape, as today. No warning.
- **JIT cache.** The facts read become part of what a JIT cache entry depends on, like the default
  library path already is (`jitcache.cpp`). Otherwise a rebuilt library would leave stale
  promotions in cached objects.

### 5.3 Using the facts

In `StackEscapeFacts`:
- **A direct call** (`ts.SymbolCallInternal`) to a declared function (no body) whose symbol has an
  imported fact: the argument at a listed index does not escape through the call. Today
  `argumentEscapes` returns `unknown-callee` there.
- **A call through a `ts.SymbolRef`** of such a function: the same.
- **A virtual call on a library class stays an escape in version 1 (my call).** A static
  library or an object file given to the link can subclass a library class, and MLIRGen cannot see
  those inputs. Trusting such calls needs a way to detect them, which is left for later.
- Everything else is unchanged: an unlisted parameter, an unknown function, a call through a value.

### 5.3a When no facts are read (my call)

`--emit=obj`, `--emit=llvm` and `--emit=bc` do not link, so the compiler cannot know which library
the object will be linked with (the CMake template and the default library's own build work this
way). They read no facts. Facts are read only when tslang itself links or loads the library. The
choice of file - lib or dll, debug or release, Android - is made today in `exe.cpp` and `jit.cpp`,
after MLIRGen. The builder moves that choice to a function MLIRGen can call, so both read the same
answer.

### 5.4 Reading from `-shared` imports too (my call)

The reading code runs on the default library and on every shared library the program imports. The
code path already exists for `__tsown_`. And the tests need a library whose facts they control: the
runner builds every test with `--no-default-lib`, so a default-library-only reader could not be
tested in the ordinary suite. Enabling `-shared` libraries costs nothing beyond this; turning it off
would cost a test-only switch.

## 6. Soundness

- A fact is a property of the library's body, computed by the analysis that already promotes
  objects inside that library. It needs no closed world. If any caller anywhere passes a value the
  body keeps, the parameter is already an escape.
- Inside the library, calls through function values, unknown callees and exported classes' virtual
  methods are already escapes, so a library function that hands its argument to a user callback
  never claims to borrow it.
- The only new trust is that the binary read is the binary linked (§5.2).

## 7. Testing

- **Producer:** a library module with functions that borrow, keep, return and hand to a callback
  their parameters. Check its emitted symbol text, by reading the built binary or by a dump flag of
  the builder's choosing. Every case gets a positive and a negative.
- **Consumer:** a program importing that library as `-shared`, with a decision test per case: the
  `--stack-promotion-report` remark says `stack` for a borrowed argument and `heap: argument` or
  `heap: unknown-callee` otherwise. Run tests under gc, rc, own and none.
- **Stale or missing facts:** a program against a library built without facts (an old build, or
  the symbol stripped) still compiles and promotes nothing through those calls.
- **Virtual-call rule:** a library class with a virtual method; one program importing only that
  library, and one also importing a second library that subclasses it. Expected: stack in the
  first, heap in the second.
- **Default library end to end:** a program using `new RegExp` with `replace`, compiled with the
  default library (outside the runner's `--no-default-lib` suite, as the default-library tests
  already are). Check the remark and the output.
- **Teeth:** make the producer list every parameter, and confirm the keeper test then reads a
  dangling pointer.

## 8. Left to the builder

- The symbol's name and text format.
- How `Dump` reads a static archive.
- Where in the pipeline the facts are emitted (the stack promotion pass, or a pass of its own
  sharing `StackEscapeFacts`).
- The JIT cache key detail.
- The default-library build changes, if its build does not already export the new symbol.
- Phasing into PRs, including the DefaultLib repository.
