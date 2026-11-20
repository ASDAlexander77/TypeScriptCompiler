# Class Member Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each class a by-name member table and use it for `c[k]` / `k in c` with run-time keys, `any[k]` on class instances (actual class), `C[k]` on class values, and a load-time layout check for `@dllimport` libraries.

**Architecture:** MLIRGen emits `<class>.members` / `<class>.static_members` as constant tuple globals (offsets via the null-GEP trick already used for interface vtables). The tables are reached from a new class-vtable header slot (actual class) and a new type-descriptor slot `TYPE_DESCR_MEMBERS`. A new op `ts.MemberKey` is lowered to a call of one shared LLVM function per module, `ts.member_key`, which looks the key up and boxes, tests or writes. `keyof C` keys don't need the table: they become a compile-time switch over the finite name set.

**Tech Stack:** C++ / MLIR (TypeScript dialect) / LLVM lowering; tests are `.ts` programs run by `test-runner` through ctest.

**Spec:** `docs/superpowers/specs/2026-11-20-class-member-table-design.md`. Read it first.

**Base:** branch from `origin/main` (`git checkout --no-track -b class-member-table origin/main`, then `git push -u`). The local checkout's branch is older than main. `TYPE_DESCR_KEYS`, `mlirGenAnyKey` and `makeAnyKeysAdapter` exist only on main. Commit the spec on this branch too.

## Deviation from the spec (implementation detail, same behaviour)

- `c[k]` with `k: keyof C` (or a literal union) is compiled as a switch over the possible names, using ordinary property access. It does not walk the table. It returns the same unboxed union, and virtual dispatch and `-mm=own` work without boxing. The table serves `string` keys, `k in c`, `any[k]` and the import check.

## Global Constraints

- Type-descriptor slots: add only, never reorder. The new slot goes right before `TYPE_DESCR_BLOCK_HEADER`, which stays last (`Defines.h` "Runtime type descriptor").
- The class vtable gets one header slot right after `.instanceOf`, on root non-`@cpp` classes only. `@value`, static and `@cpp` classes get none.
- Objects must not grow. Tables hang off the vtable and descriptor only.
- A table is made only when the class is exported, boxed into `any`, or read by a run-time key in the module.
- Private members are never in a table. Protected entries carry the access flag, and a run-time key never matches them.
- `-mm=own`: `keyof` reads of fields only. A key union that includes a method is a compile error. Anything that boxes into `any` keeps today's clear error.
- Writes: only with `keyof C` keys. `v` must be assignable to the intersection of the named fields' types (compile time). A readonly field or a method in the key set is a compile error.
- Library check message, verbatim shape: `library '<path>': <Class>.<member> <what changed>; rebuild this program against the new library`. Exit code non-zero.
- Every test prints `done.` at the end. Add each to `TSLANG_CORPUS` (rc/none sweep) unless own- or JIT-specific.
- `ctest` always needs `-C Release`. Edits to `TypeScriptOps.td` build at `-j 4` at most.
- Layout change: once the branch builds, rebuild the default library (`build.bat` with the `TSLANG_LIB_PATH`-style env cleared, every `-Model`) and the wasm/Android archives, and check the archive timestamps. Open a matching DefaultLib PR.

## Review Focus

1. A `D` held as `C`, with `k: string` naming a field only `D` has: `c[k]` gives `D`'s value (the actual class). The test is in Task 4.
2. An optional field that is unset: `k in c` is false and `c[k]` is `undefined`, both through `any` and through a `string` key. The test is in Task 4.
3. A key that is a number (`c[0]`) or a non-string `any` on a class: it reads as the name `"0"`, matching the existing `any[k]` behaviour. The test is in Task 4.
4. A class with no own fields that extends a library class: the importer's size check catches a changed base size. The test is in Task 7.
5. A class that is never read by key and never boxed: no `.members` global appears in the IR. The test is in Task 1.

---

### Task 1: Emit member tables

**Files:**
- Create: `tslang/lib/TypeScript/MLIRGenMemberTable.cpp`, with the declarations in `tslang/lib/TypeScript/MLIRGenImpl.h` near `makePendingAnyKeysAdapters` (~:12851). Add the file to `tslang/lib/TypeScript/CMakeLists.txt`.
- Modify: `tslang/include/TypeScript/Defines.h`: add `MEMBERS_NAME ".members"`, `STATIC_MEMBERS_NAME ".static_members"`, the `MEMBER_KIND_*` and `MEMBER_ACCESS_*` constants, and `MEMBERS_MARKER "// @members"`.
- Modify: `tslang/lib/TypeScript/MLIRGenModule.cpp:~1502/1559`, next to `makePendingAnyKeysAdapters`, to call `makePendingMemberTables`.
- Test: `tslang/test/tester/tests/00class_members_table.ts`, plus an IR check test in the same CMake style as the existing `--emit=mlir` FileCheck tests. If none exist, use a `PASS_REGULAR_EXPRESSION` on `--emit=llvm` output.

**Interfaces:**
- Produces:
  - `void MLIRGenImpl::ensureMemberTable(mlir::Location, ClassInfo::TypePtr, const GenContext &)`, which queues the class like `ensureAnyKeysAdapter`.
  - `void makePendingMemberTables(const GenContext &)`.
  - The global `<class.fullName>.members`, and `<fullName>.static_members` when the class has public or protected statics.
  - Entry tuple: `{name: string, kind: u8, access: u8, readonly: u8, slot: index, type: opaque (type tag), size: index, typeHash: u64}`.
  - Header: `{count: u32, flags: u32, rtti: opaque, entries: ptr}`.
  - Entries are sorted by name (byte order).
  - `slot` holds:
    - a field: its offset, tagged with `INTERFACE_OPTIONAL_STORAGE_TAG` when it's optional;
    - an accessor: its cell, tagged with `INTERFACE_ACCESSOR_CELL_TAG`;
    - a virtual method: its vtable index;
    - a non-virtual or `@final` method: its function address;
    - a static: the global's address.
  - `type` is the member's type tag (what a box stores in `ANY_TYPE`). For a method it's the bound function type's tag. `size` is the stored size.
  - `typeHash = MLIRHelper::hashOfType(declared type)`.
  - `flags` bit 0 means "has dynamic subclasses". Nothing reads it yet. Leave it at 0.
- Triggers that call `ensureMemberTable`:
  - the class is exported (`isExport`) at `mlirGenClassVirtualTableDefinition`;
  - a class instance is boxed into `any` (the same place `ensureAnyKeysAdapter` is called, `MLIRGenCast.cpp:1391`);
  - Tasks 4 and 5 call it from element access and `in`.

- [ ] **Step 1: Write the failing test.** In `00class_members_table.ts`, put two classes:
  - `class P { x = 1; protected y = 2; private z = 3; get g() { return 4; } m() { return 5; } static s = 6; }`, boxed into `any` once;
  - `class Q { a = 1 }`, never boxed.

  The IR check expects these exact global names: `P.members` present, `P.static_members` present, `Q.members` absent. The program itself prints `done.`.
- [ ] **Step 2: Run it and check it fails.** Command: `ctest -C Release -R class-members-table` in `__build/tslang/windows-msbuild-2026-release`. Expected: FAIL, because `P.members` is not found.
- [ ] **Step 3: Implement `ensureMemberTable` and `makePendingMemberTables`** in `MLIRGenMemberTable.cpp`.
  - Collect public and protected members, recursing into the bases (like `anyKeysOfType`, `MLIRGenCast.cpp:4701`). Skip private members, names starting with `.` or `#`, and `constructor`.
  - Compute offsets the way `mlirGenClassVirtualTableDefinitionForInterface` does (`MLIRGenClasses.cpp:~4909-4946`): `NullOp` cast to the class, then property access, then `GetReferenceFromValue`, then cast to index.
  - Get each member's type tag with the existing op the box uses for its tag.
  - Make the globals with `createGlobalVariable`, using the same comdat and export flags as `mlirGenClassSizeStaticField` (`MLIRGenClasses.cpp:3605`). Export them only when the class is exported.
- [ ] **Step 4: Run the test and check it passes.** Same command. Expected: PASS.
- [ ] **Step 5: Run the full Release ctest.** Command: `ctest -j 8 -C Release`. Expected: same pass count as `origin/main`. Nothing reads the tables yet.
- [ ] **Step 6: Commit** with the message "Emit per-class member tables (.members / .static_members)" and the spec doc included.

### Task 2: Reach the table at run time (vtable header slot and descriptor slot)

**Files:**
- Modify: `tslang/include/TypeScript/MLIRLogic/MLIRGenStore.h:~1090`. Push a `.members` record right after `.instanceOf`, root class and non-`@cpp` only.
- Modify: `tslang/lib/TypeScript/MLIRGenImpl.h:16581` (`getVirtualTableType`): the new record kind is Opaque.
- Modify: `tslang/lib/TypeScript/MLIRGenClasses.cpp:~5130` (emission loop): store the address of `<fullName>.members`, or null when the class has no table. A subclass's vtable stores its own table.
- Modify: `Defines.h`. Add `CLASS_VTABLE_MEMBERS_SLOT 1` and `TYPE_DESCR_MEMBERS 7`, and bump `TYPE_DESCR_BLOCK_HEADER` to 8. Update the comment.
- Modify: `TypeDescriptorLogic.h:39-45,103-110`: add the record field and a `getMembersFromTag`.
- Modify: `LLVMCodeHelper.h:359-432` (`getOrCreateTypeDescriptorName`): add a parameter.
- Modify: `LowerToLLVM.cpp:~498-513`: pass the table address from a new module attribute `ts.member_tables` (type hash → global name). `makePendingMemberTables` writes that attribute the same way `ANY_KEYS_ADAPTERS_ATTR_NAME` is written.
- Test: the whole existing suite. The vtable shift is the risk here.

**Interfaces:**
- Produces:
  - `CLASS_VTABLE_MEMBERS_SLOT` (1): the slot holding the actual class's table, or null.
  - `TYPE_DESCR_MEMBERS`: the box tag's static-type table, or null.
  - `ts.member_tables` module DictionaryAttr.

- [ ] **Step 1:** Add the slot and record. Build at `-j 4`. Command: `cmake --build __build/tslang/windows-msbuild-2026-release --config Release -j 4`. Expected: the build succeeds.
- [ ] **Step 2: Run the full Release ctest.** Expected: same pass count as Task 1. Pay particular attention to the `shared-class-members` import tests, the `@cpp` tests and `OwnershipSignatureResolve`'s `.instanceOf` check (`:697`, still index 0).
- [ ] **Step 3: Rebuild the default library** for every model and the wasm/Android archives (Global Constraints). Then rerun the suite with the `TSLANG_LIB_PATH` env pointing at the new library. Expected: green.
- [ ] **Step 4: Commit** with the message "Class vtable and type descriptor point at the member table (layout change)".

### Task 3: `ts.MemberKey` op and its shared run-time function

**Files:**
- Modify: `tslang/include/TypeScript/TypeScriptOps.td`, next to `AnyKeysAdapterOp` (:516). Build at `-j 4`.
- Create: `tslang/include/TypeScript/LowerToLLVM/MemberTableLogic.h`, which builds the LLVM function `ts.member_key` once per module, linkonce_odr.
- Modify: `tslang/lib/TypeScript/LowerToLLVM.cpp`: add `MemberKeyOpLowering`.
- Test: covered through Task 4's tests. This task's check is that the build passes and the existing suite stays green.

**Interfaces:**
- Produces `ts.MemberKey(table: opaque, object: opaque, op: i32, key: string, value: any) -> any`. The ops are the existing `ANY_KEYS_OP_GET/HAS/SET`.
- Semantics of `ts.member_key`:
  - Binary-search the table by name. Only public entries match.
  - Not found: GET returns `undefined`, HAS returns `false`, SET throws a TypeError with the same text the adapters use today.
  - Optional field: its flag is read through the tagged slot. When unset, the field counts as absent.
  - Field read: box `size` bytes from `object + offset` with the entry's tag, as `AnyLogic::castToAny` does. Retain through the tag's `TYPE_DESCR_RETAIN`.
  - Accessor: call the getter cell with `object` as `this`, then box the result.
  - Method: read the address from the object's vtable at the slot index, or take the stored address. Box a bound function `{object, fn}` with the entry's tag.
  - SET on a field: check that `value`'s tag equals the entry's tag, then copy. Release the old value through `TYPE_DESCR_RELEASE` and retain the new one. A readonly field, a getter-only accessor or a method throws a TypeError (same texts as now).
- Allocation goes through the same allocator call `AnyLogic::castToAny` uses, so every memory model is respected.

- [ ] **Step 1:** Add the op and an empty lowering that emits the call. Build at `-j 4`. Expected: the build succeeds.
- [ ] **Step 2:** Implement `MemberTableLogic::getOrCreateMemberKeyFunction(ModuleOp) -> LLVM::LLVMFuncOp` with the semantics above.
- [ ] **Step 3: Run the full Release ctest.** Expected: green. Nothing uses the op yet.
- [ ] **Step 4: Commit** with the message "ts.MemberKey: look up, box and write a member through a class's table".

### Task 4: `any[k]` and `string` keys on class instances use the actual class's table

**Files:**
- Modify: `tslang/lib/TypeScript/MLIRGenCast.cpp:5109` (`mlirGenAnyKey`). When the box holds a class (a run-time tag check through `TYPE_DESCR_MEMBERS` being non-null), read the actual class's table from the object's vtable slot `CLASS_VTABLE_MEMBERS_SLOT`. If that's null, use the descriptor's table. Then emit `ts.MemberKey`. Keep `makeAnyKeysAdapter` for tuples, interfaces and arrays, and stop making it for classes.
- Modify: `tslang/lib/TypeScript/MLIRGenAccessCall.cpp:1675-1695`. A class with no string indexer, a key that's not a constant and not a `keyof` union: call `ensureMemberTable`, then emit `ts.MemberKey(GET)`. The result type is `any`. Under `-mm=own` this is an error with the text `reading a class by a string key gives any, which -mm=own does not allow: use a keyof key`.
- Modify: `tslang/lib/TypeScript/MLIRGenImpl.h:7313-7327` (`mlirGenInLogic`): a non-constant key on a class goes to `ts.MemberKey(HAS)`.
- Test: `tests/00class_string_keys.ts`. Update `tests/00any_keys.ts` where a method read is now a bound function instead of a TypeError.

**Interfaces:** consumes Task 1's `ensureMemberTable`, Task 2's slots and Task 3's op.

- [ ] **Step 1: Write the failing test** `00class_string_keys.ts`:
  ```ts
  class C { a = 1; o?: number; m() { return this.a + 10; } }
  class D extends C { d = "dee"; override m() { return 99; } }
  function key(s: string) { return s; }   // a run-time string
  const c: C = new D();
  assert(<string>c[key("d")] == "dee");          // actual class (Review Focus 1)
  assert(<number>c[key("a")] == 1);
  assert(typeof c[key("nope")] == "undefined");
  assert(!(key("o") in c) && typeof c[key("o")] == "undefined"); // unset optional (Focus 2)
  c.o = 5; assert(key("o") in c);
  const f = <() => number>c[key("m")]; assert(f() == 99);        // bound, virtual
  const x: any = c; assert(<string>x[key("d")] == "dee");
  class N { "0" = 7 }  const n: any = new N(); assert(<number>n[0] == 7); // Focus 3
  print("done.");
  ```
  Register `test-compile-00-class-string-keys` and `test-jit-00-class-string-keys`, and add the file to `TSLANG_CORPUS`. Add an own error test `test-compile-own-00-class-string-key-error`, with `PASS_REGULAR_EXPRESSION` set to the own message above and the file in `verify-ownership.cmake`'s `not_compilable_alone`.
- [ ] **Step 2: Run it and check it fails.** Command: `ctest -C Release -R class-string-key`. Expected: FAIL (an indexer-not-declared error, or `d` undefined).
- [ ] **Step 3: Implement** the three changes listed under Files.
- [ ] **Step 4: Run the tests and check they pass.** Run the new tests and `-R any-keys`. Expected: PASS.
- [ ] **Step 5: Run the full Release ctest.** Expected: green.
- [ ] **Step 6: Commit** with the message "c[k], k in c and any[k] on a class read the actual class's member table".

### Task 5: `keyof C` keys: reads, writes, `k in c`, and statics

**Files:**
- Modify: `tslang/include/TypeScript/MLIRLogic/MLIRTypeHelper.h:~2827` (the class branch of `getFields` as used by `getKeyOf`, `MLIRGenTypes.cpp:1928`). `keyof C` must be the public fields, accessors and methods, recursing into bases, with no compiler names (`.vtable` and others). Change this only for `getKeyOf`. Leave the other `getFields` callers alone: add a `getPublicMemberNames(ClassType)` helper and call it from `getKeyOf`.
- Modify: `MLIRGenAccessCall.cpp:1675-1695`. When the key's type is a union of string literals, or `keyof C` resolved to one, emit a switch: compare the key with each name and read `c.name` through normal property access. The result type is the union of the member types, with a method as its bound function type. This applies to class values too (`C[k]` with `k: keyof typeof C`, through static access).
- Modify: the element-access assignment path. A write with such a key emits a switch of stores. Check at compile time that the value is assignable to the intersection of the fields' types. Error texts:
  - `'<name>' is readonly and can't be written through a key`
  - `a method can't be written through a key`
- Modify: `-mm=own`. If a method is in the key set: error `reading a method through a key moves the object under -mm=own; the key set includes '<name>'`.
- Test: `tests/00class_keyof_keys.ts`, plus the error tests `00class_keyof_write_readonly_error.ts` and `00own_class_keyof_method_error.ts`.

- [ ] **Step 1: Write the failing tests.** `00class_keyof_keys.ts`:
  ```ts
  class C { a = 1; b = "x"; readonly r = 2; m() { return 3; } static s = 4; static t = "y"; }
  function pick(i: number): keyof C { return i == 0 ? "a" : i == 1 ? "b" : "m"; }
  const c = new C();
  const v = c[pick(0)]; assert(<number>v == 1);
  assert(<string>c[pick(1)] == "x");
  assert((<() => number>c[pick(2)])() == 3);
  class W { a = 1; a2 = 2; }  const ww = new W();
  const wk: "a" | "a2" = pick(0) == "a" ? "a2" : "a"; ww[wk] = 7; assert(ww.a2 == 7);
  const sk: keyof typeof C = "t"; assert(<string>C[sk] == "y");
  print("done.");
  ```
  Error tests, each registered with `PASS_REGULAR_EXPRESSION` and listed in `not_compilable_alone`:
  - `00class_keyof_write_intersection_error.ts`: `c[w] = "z"` with `w: "a" | "b"`. Expect `not assignable`.
  - `00class_keyof_write_readonly_error.ts`: a write through a key of `"a" | "r"`. Expect the readonly message.
  - `00own_class_keyof_method_error.ts` (`-mm=own`): `c[pick(2)]`. Expect the own method message.
- [ ] **Step 2: Run them and check they fail.** Command: `ctest -C Release -R class-keyof`. Expected: FAIL.
- [ ] **Step 3: Implement** the changes listed under Files.
- [ ] **Step 4: Run the tests and check they pass.** Run the new tests plus every existing test that uses `keyof` (`ctest -C Release -R keyof`). Expected: PASS. A `keyof` result on classes now includes methods. Fix any test that relied on fields only.
- [ ] **Step 5: Run the full Release ctest and the own suite** (`-R own`). Expected: green.
- [ ] **Step 6: Commit** with the message "keyof C keys: a switch over member names for reads, writes and statics".

### Task 6: Libraries export their tables and mark their declarations

**Files:**
- Modify: `tslang/lib/TypeScript/DeclarationPrinter.cpp`. When the module's classes have tables, print `MEMBERS_MARKER` as the first line of each `__decls_<module>`.
- Modify: `MLIRGenMemberTable.cpp`: an exported class always gets its table (Task 1's trigger), exported by symbol.
- Test: extend the `shared-class-members/` import tests (`CMakeLists.txt:6405`), checking that the library exports `<Class>.members`. The program reads a library object by a `string` key and checks the value.

- [ ] **Step 1: Write the failing test.** Add to the `shared-class-members` program: `const k: string = ...; assert(<number>(obj as any)[k] == <expected>); print("done.")`. Expected: FAIL in the jit-shared variant if the table isn't exported.
- [ ] **Step 2: Implement** the marker and the export.
- [ ] **Step 3: Run the tests and check they pass.** Command: `ctest -C Release -R shared-class-members`. Expected: PASS for all three variants.
- [ ] **Step 4: Commit** with the message "Libraries export class member tables and mark their declarations".

### Task 7: Check the layout when a `@dllimport` library loads

**Files:**
- Modify: `tslang/lib/TypeScript/MLIRGenModule.cpp:1907` (`mlirGenImportSharedLib`).
  - If the declarations lack `MEMBERS_MARKER`, warn: `library '<path>' predates layout checks; field layout is not verified`.
  - Otherwise, after `LoadLibraryPermanentlyOp` in the ctor (:2038-2081), emit for each imported class a constant "expected" table in the same entry format (Task 1). It covers the fields the importer reads at fixed offsets: every public and protected field of the class and its library bases. Add the instance `.size` the importer assumes. Then emit one call to `ts.check_members(expected, SearchForAddressOfSymbol("<Class>.members"), libraryPath, className)`.
- Create: the check function, generated in `MemberTableLogic.h` like `ts.member_key`.
  - For each expected entry, find it by name. Compare `kind`, `slot` (offset) and `typeHash`.
  - Compare the library's `<Class>.size` with the expected size.
  - On the first mismatch, print the message from Global Constraints and call `exit(1)`. `<what changed>` is one of:
    - `is missing`
    - `moved from offset N to M`
    - `changed type`
    - `changed size from N to M bytes`
  - A missing `<Class>.members` symbol: print `library '<path>': <Class> has no member table; rebuild the library` and exit.
- Test: a new import test directory `tslang/test/tester/tests/shared-layout-check/`, registered with `tslang_add_import_tests`:
  - `lib_v1.ts`: `export class Box { a = 1; b = 2 }`
  - `lib_v2.ts`: `export class Box { z = 0; a = 1; b = 2 }`
  - `program.ts`: `@dllimport` of the library, reads `box.b`, prints `done.`

  The library v1 run passes. A second test builds the program against v1 and runs it with v2's DLL in its place, expecting `PASS_REGULAR_EXPRESSION "Box.b moved from offset"`. Add a third case for Review Focus 4: `class Mine extends Box {}` in the program, with v2 adding a private field at the end. Expect `changed size`.

- [ ] **Step 1: Write the failing tests** above. Use a small CMake script, modelled on the existing shared-lib tests (`CMakeLists.txt:1864-1885`), that compiles both versions and swaps the DLL. Expected: FAIL. The program runs and prints a wrong value, or crashes.
- [ ] **Step 2: Implement** the marker check, the expected table and `ts.check_members`.
- [ ] **Step 3: Run the tests and check they pass.** Command: `ctest -C Release -R shared-layout-check`. Expected: PASS for all cases.
- [ ] **Step 4: Run the full Release ctest.** Expected: green.
- [ ] **Step 5: Commit** with the message "Check a @dllimport library's class layout when it loads".

### Task 8: Docs, default library and the final run

**Files:**
- Modify: `docs/whats-new.md`. Add one entry covering `c[k]`, `k in c`, methods through `any[k]` and the library layout check. Note that `keyof C` now includes methods.
- Modify: `README` only if the index needs a link (it's an index, per #845).
- DefaultLib repo: rebuild the archives and open a PR (layout change).

- [ ] **Step 1:** Rebuild the default library (debug and release, every model) from a clean DL clone. Run `tests.ps1` with the new compiler. Expected: green.
- [ ] **Step 2: Run the full Release ctest** with the library env set, and the debug default-library check as `create-release` does it. Expected: green.
- [ ] **Step 3: Measure the size change** of the default-library archives and of three test executables, against `origin/main`. Put the numbers in the PR description.
- [ ] **Step 4: Commit** the docs and open the PRs (compiler and DefaultLib).
