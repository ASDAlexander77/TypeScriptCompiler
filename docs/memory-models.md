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

## Other limits of `-mm=rc`

- **Objects crossing between differently-managed modules are never freed.** If you link a
  module built `-mm=rc` against one built `-mm=gc` — including the standard library, which is
  built with garbage collection — anything allocated on the other side leaks rather than being
  freed twice. The compiler warns when it can see the mismatch. Building everything with the
  same `-mm=` avoids it.
- **Counts are not atomic.** `-mm=rc` is single-threaded today.

## Shared libraries and `-mm=gc`

**A program that loads a tslang shared library must link Boehm as a DLL, not statically.**

If the executable and the library each link `gc.lib` statically, each gets its own collector,
with its own heap and its own idea of what the roots are. The library's collector does not scan
the executable's roots, so it frees objects the executable is still holding. The symptom is not
a crash: the freed memory is reallocated and the program reads a plausible wrong value, which
only shows up when what was written over it differs from what was there.

Build the shared collector with `scripts/build_gc_release_shared_vs.bat`, link against
`3rdParty/gcdll/x64/release/lib/gc.lib`, and ship `gc.dll` beside the executable.

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
