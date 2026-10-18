# Memory models

The compiler can manage heap memory in three ways, selected with `-mm=`:

| flag | what it does | when to use it |
| --- | --- | --- |
| `-mm=gc` | **Garbage collection** (Boehm). The default. | Almost always. Nothing to think about, and nothing leaks. |
| `-mm=rc` | **Reference counting.** Objects are freed the moment the last reference to them goes away. No collector, no pauses, no runtime dependency on libgc. | Predictable latency, WebAssembly, or shipping without libgc — **provided you read the cycles section below.** |
| `-mm=none` | **Nothing is ever freed.** | Short-lived programs where the process exits before memory matters. |

`-mm=gc` is the default and stays the default. Everything below is about what changes if you
choose `-mm=rc`.

```bash
tslang --emit=exe -mm=rc hello.ts
```

## What `-mm=rc` gives you

Memory is reclaimed deterministically, at the point the last reference is dropped, rather than
whenever a collector next runs. On allocation-heavy programs it holds close to the working set
rather than to the total allocated:

| program | `-mm=gc` | `-mm=rc` | `-mm=none` |
| --- | --- | --- | --- |
| a ray tracer, 256x256 | 5.8 MB | 4.1 MB | 114.5 MB |
| the same, 512x512 | 5.5 MB | 4.1 MB | 445.0 MB |
| an n-body simulation | 5.6 MB | 4.1 MB | 4.1 MB |

Peak working set, ahead-of-time build, `--opt --opt_level=3`. The ray tracer's `-mm=rc` figure is
flat across a 64-fold change in image size, because nothing accumulates.

There is no collector thread, no pause, and no `libgc` to ship.

## Reference cycles are not collected

**This is the one thing to know before choosing `-mm=rc`.** If two objects refer to each other,
directly or through a chain, neither one's count ever reaches zero and neither is ever freed:

```typescript
class Node {
    parent: Node;
    name: string;
    constructor(name: string) { this.name = name; }
}

const a = new Node("a");
const b = new Node("b");
a.parent = b;
b.parent = a;      // a cycle: neither a nor b will ever be freed under -mm=rc
```

Measured, that costs everything: a loop building one such pair per iteration holds **22.6 MB**
under `-mm=rc`, exactly what `-mm=none` holds — reference counting reclaims none of it. Break
the cycle by removing one of the two assignments and the same loop holds 4.1 MB.

This is a **defined property of the mode, not a bug**, and it is the same trade Swift makes with
ARC. The difference is that here it is opt-in: `-mm=gc` is the default, handles cycles without
you thinking about it, and is one flag away.

### Shapes that leak

Anything that refers back to itself, however indirectly. Each of these was measured, and each
holds exactly as much as `-mm=none` — that is, reference counting reclaims none of it:

- **Parent/child links** — `class Node { parent: Node; children: Node[] }`, the example above.
- **Mutually referencing objects** — `a.peer = b; b.peer = a`.
- **Doubly linked lists** — every node holds its neighbour and is held by it.
- **An object holding a callback that captures the object** — the closure's capture box holds
  the object, and the object holds the closure.

### Shapes that do not leak

- **Trees and lists with no back-references** — the common case.
- **Strings**, and arrays of them. A string never points at another heap object, so no cycle
  involving one can exist.
- **Self-recursive functions.** A named recursive function is not a cycle — it holds no
  reference to itself at run time. (A self-referential *arrow function* would be, but the
  compiler does not currently accept one.)

### What to do about it

1. **Use `-mm=gc`** — the default — if your data has cycles and you do not want to think about
   them. This is the right answer for most programs.
2. **Break the cycle by hand** where you know about it: null out the back-reference when you are
   done with the structure, or store a key/index instead of a pointer back to the owner.
3. If neither fits, `-mm=rc` is not the right mode for that program.

A `WeakRef<T>` that lets you declare a back-reference as non-owning is designed but not
implemented; see `tslang/docs/reference-counting-evaluation.md` §9.8. When it lands it will be
the fourth option here, and it does not change anything above.

## The standard library

The standard library is built once per memory model, and your program links the build matching
its own `-mm=`. Nothing to configure — the compiler picks it.

It has to work that way: the library allocates the way the model it was built for allocates.
The `-mm=gc` build calls into Boehm and brings `libgc` with it; the `-mm=rc` build maintains
reference counts and brings no collector at all. A hello-world is 335 KB under `-mm=gc` and
145 KB under `-mm=rc` for exactly that reason.

If the build for your model is missing, the compiler says so and names the directory rather
than falling back to another model's copy — that would link and then misbehave at run time.
Build them with the default library's `build.bat` (`build.sh` on Linux), which produces all
three.

## Stack promotion

Under every model, an object that provably never outlives the function that made it is put in
that function's stack frame instead of on the heap. This happens automatically, and you never
see it: the program does exactly what it did before, only with fewer heap allocations. `new C()`
and `C()` are the same thing, and either may end up on the stack.

The compiler promotes:

- **class instances** and **object literals with methods**, when the object is only read and
  written through its fields, has its methods called, is compared, or is passed to functions in
  the same module that do not keep it;
- **array literals** that never change length (no `push`, `pop`, `splice`, `length =`) and are
  never handed somewhere that might change or keep them; under `-mm=rc` and `-mm=own`, only when
  the elements own no memory (numbers, booleans);
- a **closure's capture box**, when the closure never leaves the function; under `-mm=gc` and
  `-mm=none` only.

Anything the compiler cannot prove safe stays on the heap, silently. In this version that
includes an object that is:

- returned, thrown, stored in a field, array, global or captured variable, or converted to
  `any`, an interface or a union;
- passed to a function in another module, including the default library;
- made in a function that is `async`, a generator, or recursive, or - under `-mm=rc` and
  `-mm=own` - one that has a `try`;
- still in use when the next iteration of its loop makes a new one. A use on the way out of the
  loop - a `return` or a `break` out of a search - is not.

Under `-mm=rc` and `-mm=own`, an object that owns memory stays on the heap in a module that uses
`Shared<T>`, because `Shared.count` would see its contents given back later.

Two limits keep frames small: one object may take at most 256 bytes
(`--stack-promotion-max-object=`), and one function at most 4096 bytes in all
(`--stack-promotion-max-frame=`). These are estimates, made before the exact layout is known.

`--no-stack-promotion` turns the feature off. `--stack-promotion-report` prints, for every
allocation, whether it went on the stack and, if not, why:

```
p.ts:11:19: remark: stack-promotion: stack
p.ts:8:15: remark: stack-promotion: heap: returned
```

Measured, best of three, ahead-of-time build with `--opt --opt_level=3`, in milliseconds. Each
cell is `--no-stack-promotion` → default:

| program | `-mm=gc` | `-mm=rc` | `-mm=own` | `-mm=none` |
| --- | --- | --- | --- | --- |
| vector temporaries, 5 million iterations | 18 → 20 | 296 → 16 | 287 → 27 | 27 → 20 |
| ray tracer | 342 → 341 | 153 → 170 | — | 182 → 157 |

The gain is where a heap allocation is expensive: under `-mm=rc` and `-mm=own`, an object made
and dropped in a loop costs a `malloc` and a `free` per iteration, and promotion removes both.
Under `-mm=gc` the optimiser already deletes an allocation nothing reads, so there is little left
to gain. `—`: the program does not compile under `-mm=own`.

The ray tracer under `-mm=rc` is slower with promotion, although only two objects in its one-time
scene setup are promoted and the code of its hot loop is identical either way. Promotion costs
nothing here: the two objects were the program's first heap allocations, and without them every
later object lands somewhere else on the heap, which this program's speed happens to depend on. Put
two throwaway heap allocations of the same size back at the same point and the promoted build runs
at the unpromoted speed (157 ms against 158). Expect the same kind of swing, in either direction,
from any change to what a program allocates early.

## Other limits of `-mm=rc`

- **Objects crossing between differently-managed modules are never freed.** If you link a
  module built `-mm=rc` against one built `-mm=gc`, anything allocated on the other side leaks
  rather than being freed twice. The compiler warns when it can see the mismatch. Building
  everything with the same `-mm=` avoids it. (The standard library is not affected — see
  above.)
- **Counts are not atomic.** `-mm=rc` is single-threaded today.

## Shared libraries and `-mm=gc`

**A program that loads a tslang shared library must link Boehm as a DLL, not statically.**
The same rule is why, on Windows, `TypeScriptRuntime.dll` (the JIT's runtime) and the default
library's `TypeScriptDefaultLib.dll` take the collector from `gc.dll`: under the JIT, or beside a
user's shared library, they share a process with other `gc` code, and one static collector
among them frees what the others hold. `gc.dll` ships beside `tslang.exe`.

If the executable and the library each link `gc.lib` statically, each gets its own collector,
with its own heap and its own idea of what the roots are. The library's collector does not scan
the executable's roots, so it frees objects the executable is still holding. The symptom is not
a crash: the freed memory is reallocated and the program reads a plausible wrong value, which
only shows up when what was written over it differs from what was there.

On Windows `tslang` makes the choice itself:

- `--emit=dll`, and `--emit=exe` for a program that imports a tslang shared library, link the
  shared collector and copy `gc.dll` beside the output.
- `--emit=exe` for a program that imports none keeps the static `gc.lib`, and ships as one file.

It finds the shared collector through `--gc-shared-lib-path` (or `GC_SHARED_LIB_PATH`), else the
`gcdll` folder inside `--gc-lib-path` — which is where the Windows release package ships it, beside
the static `gc.lib` at its root; both files are named `gc.lib`, so the folder is what tells them
apart — else `--gc-lib-path` itself when that already names a shared build. If none of those has
one, the build stops with an error rather than linking a collector of its own.

From a source build, the shared collector comes from `scripts/build_gc_release_shared_vs.bat`:
`--gc-shared-lib-path=3rdParty/gcdll/x64/release/lib` (with `gc.dll` in its `../bin`).

Linking by hand (`--emit=obj` and your own linker) makes no choice for you: link the executable
**and** every shared library against the shared `gc.lib`, and ship `gc.dll` beside the executable.

What to ship with a Windows program that loads a tslang shared library, all in one folder:

- the program and its shared libraries,
- `gc.dll` (from `gcdll/` in the release package),
- `TypeScriptDefaultLib.dll` (from `defaultlib/dll/x86_64/pc/windows/msvc/<release|debug>/gc`;
  `tslang --print-default-lib-dir=dll [--di]` names it), which every shared library
  built with the default library imports.

A shared library that was linked against the static `gc.lib` anyway — by an older `tslang`, or by
hand — is refused when you import it, under `--emit=exe` and `--emit=jit` alike:

```
error: shared library 'foo.dll' links its own garbage collector (the static gc.lib). Objects
crossing between it and this module can be freed while still in use. Rebuild it with tslang
--emit=dll, which links gc.dll.
```

The compiler reads this from the library itself, so it holds however the library was linked. The
JIT checks the default library's DLL the same way before loading it.

On Linux nothing needs to ship. A program that imports a tslang shared object exports its own
collector (`--whole-archive` plus `--export-dynamic-symbol=GC_*`), and the dynamic loader binds
the shared object's `GC_*` calls to it, so the copy linked into the shared object is never used.
The default library's `.so` links no collector at all. Under the JIT, `libTypeScriptRuntime.so`
exports the collector the same way. If you link a program by hand, pass those two options.

Statically linked programs are unaffected and keep the static `gc.lib` — one binary already
means one collector. `-mm=rc` and `-mm=none` are unaffected either way: neither has a collector.

## Mixing modules

A shared library records the model it was built under, and the compiler warns when you import
one built differently:

```
warning: shared library 'foo.dll' was built with -mm=gc, this module with -mm=rc.
Objects crossing between them are never reclaimed.
```

The link is allowed and the program is correct — objects that cross simply leak, rather than
being freed by one side while the other still holds them. Build every module with the same
`-mm=` to avoid it.
