# tslang issues found while working on native_ts_graphics

Issues hit while porting https://github.com/ASDAlexander77/native_ts_graphics to tslang. Every repro
should be a complete file that was run (`tslang file.ts`, JIT, Windows x64), with the tslang version.

## Open

Items 1-20 (collected 2026-10-12 to pre-alpha108) are fixed or answered; their write-ups are in this
file's git history. The last one, 20 (an enum `E | undefined` compared with an `E` variable was always
`false`, with "Not applicable logical operator for type: 'E'"), was a missing enum case in the
comparison lowering: regression test `tslang/test/tester/tests/00enum_compare.ts`.

### 21. A C `string` result isn't copied when its `.d.ts` was first loaded by an imported module

tslang v0.0-pre-alpha111. Found 2026-10-31.

#575 (issue 6) copies the `string` a C function returns (`ts.StringCopy`), so a reused buffer or
another string's middle never becomes a program string. That copy is skipped when the
`declare function` comes from a `.d.ts` that a module this file imports also references: the
result is the raw `char*` again, with issue 6's effects. Under `-mm=gc`, stored results show the
buffer's last text. Under `-mm=rc`, retaining one writes its count into the 8 bytes before C's
pointer, and releasing it frees C's memory.

Repro: four files in one folder.

`c_decls.d.ts`:
```ts
declare function strerror(errnum: int): string;
```

`other2.ts`:
```ts
/// <reference path="c_decls.d.ts" />
export class Other {
    name(): string {
        return "other";
    }
}
```

`main2.ts`, which imports `other2.ts`:
```ts
/// <reference path="c_decls.d.ts" />
import { Other } from "./other2";

// strerror reuses one buffer: each result has to be copied when it is stored.
function main() {
    const messages: string[] = [];
    messages.push(strerror(2));
    messages.push(strerror(13));
    print(messages[0], "|", messages[1], "|", new Other().name());
}
```

`main3.ts`, the same without the import:
```ts
/// <reference path="c_decls.d.ts" />

function main() {
    const messages: string[] = [];
    messages.push(strerror(2));
    messages.push(strerror(13));
    print(messages[0], "|", messages[1]);
}
```

Run (`tslang --emit=jit -mm=gc --jit-cache=false <file>`; `-mm=rc` prints the same):
```
main2.ts -> Permission denied | Permission denied | other      (wrong: the first should be "No such file or directory")
main3.ts -> No such file or directory | Permission denied      (right)
```

The IR shows it (`--emit=llvm -mm=rc`). In `main3.ts` the call is followed by
`strlen`/`__tslang_heap_malloc`/`memcpy_s`. In `main2.ts` it is followed directly by
`call void @tsretv_...(ptr %result)`, which counts the raw pointer.

Where to look: `isForeignCall` in `tslang/lib/TypeScript/MLIRGenImpl.h` (about line 10118) answers
false unless `declaredFunctions[name].second == theModule.getOperation()`.
`MLIRGenFunctions.cpp:972` records each declaration with the module being generated when it was
seen. When the `.d.ts` was first reached through the import, the entry presumably belongs to
`other2.ts`'s pass, or this file's `/// <reference>` is skipped as already loaded, so the check
fails. A declaration's foreignness shouldn't depend on which module first loaded its `.d.ts`: a
top-level `declare function` with no body anywhere in the program is C either way.

How native_ts_graphics hit it: `core/rtxpt_game.ts` imports `./donut` (`donut.ts` references
`donut_interop.d.ts`) and calls `Donut_GetListedFile`, `Donut_JsonGetString` and
`Donut_GetNodeName` itself.
- Desktop (`gc`): stored file paths and light names aliased one buffer, until the C++ side returned a
  separate string for each value.
- Android (`-mm=rc`): the stored paths were freed at the end of `RtxptGame.sceneLoaded`, which
  crashed with "Scudo ERROR: misaligned pointer when deallocating".

The workaround there copies each result with a template literal, `` `${Donut_GetListedFile(i)}` ``
(the template's copy never retains the raw pointer). It can go once this is fixed.

Suggested regression test: `00c_string_copy.ts`'s case, split over a module and an importer that
reference the same `.d.ts`, checking that the stored `strerror` texts differ.
