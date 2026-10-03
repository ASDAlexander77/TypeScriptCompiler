# `Shared<T>` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A built-in `Shared<T>`, a counted handle to a block that holds one `T`, which `-mm=own`
leaves to rc's counting while everything else stays single-owner (own phase 8).

**Architecture:** A new dialect type `!ts.shared<T>` and three ops (`ts.SharedNew`,
`ts.SharedValueRef`, `ts.SharedCount`). `s.value` is a `ts.Load` of the `ts.SharedValueRef` place
and `s.value = x` is the field-store pattern on it, so MLIRGen's existing ownership ops and own's
existing place-read machinery apply. Retains and releases of a handle lower to rc's
`__tslang_inc_ref`/`__tslang_dec_ref` in every model that tracks ownership. The own inference
pass leaves them alone, and a borrow read through a handle is ended by any write that could reach
it through another handle and by any call that may drop.

**Tech Stack:** C++17, MLIR/LLVM (TableGen ODS), the tslang test-runner, CMake/ctest, PowerShell
(`measure.ps1`), WSL GCC for Linux.

**Spec:** `tslang/docs/superpowers/specs/2026-09-24-own-memory-model-design.md`, §23 (approved),
with §2, §15 and §22 as background.

## Global Constraints

- Branch `own-shared` (spec §23 is commit 394c8b8d, on main 8b9d6316).
- **Form (§23):** a wrapper; the count lives in the handle's block header and the block's payload
  is one `T`. `Shared<T>` is built in (it works with `--no-default-lib`). A program's own type named
  `Shared` is found first and wins.
- **Members (§23.1):** `new Shared(x)`, `s.value` read, `s.value = x`, `===`/`!==` on handles, and
  `Shared.count(s)` (the count under rc and own, `-1` under gc and none).
- **Under own (§23.2):** every retain and release of a handle stays and is counted. A handle is a
  value of type `!ts.shared<T>`, or a tag-free union or an optional of one with `null` or
  `undefined`. The pass never erases or reports one.
- **The release of a handle (§23.2):** `__tslang_dec_ref`, and at zero, release the payload `T` with
  `T`'s routine and free the block, in every model that tracks ownership (rc and own).
- **Reading through a handle (§23.3):** such a borrow ends at a write into a place reached through
  any `Shared`, or at a call that is not `__own_no_drops`. A handle copied out is a counted retain,
  not a borrow. The error is the existing "borrows a field but is used here after it may be
  released or overwritten".
- **Cycles (§23.4):** they leak; there is no `WeakRef` in this phase.
- **Never branch on the memory model in MLIRGen** (§4.3). Model differences go in the lowering and
  in the own passes.
- **Unchanged where `Shared` is not used:** the corpus (453 / 593 plain, 451 / 593 `--opt`) and its
  LLVM IR for every file that compiles.
- **Line endings:** sources are CRLF. Do not edit them with Git Bash `sed -i`, which rewrites them
  to LF. Use the Edit tool, or Python with `newline=''`.
- **Build:** `cmake --build __build/tslang/windows-msbuild-2026-release --config Release --target tslang --parallel 8`.
  - Editing a `.td` file regenerates the dialect. If MSVC fails with C1060, rebuild with
    `--parallel 3`.
  - Run tests with `ctest -C Release` (without `-C`, every test is "Not Run") and `--timeout 300`.
- **Commits** are GPG-signed. Never pass `--no-gpg-sign`. If signing times out, stage everything and
  hand the commit to the user with the message in a file. End commit messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Test rules:** never read a catch variable's value in a test. Printing a class instance needs its
  own `toString`, so print numbers and strings only. Integer literals are `s32`.

## Review Focus

1. **`Shared<T> | null` in a condition.** `while (cur)`, `if (h.s)` and `cur !== null` must narrow
   to the handle, and `null` must compare. Expected: they behave as for a class. The union is one
   pointer with no tag. Task 2's `own_shared_graph` walks such a list.
2. **A handle given to a function and returned from one.** The count across a call, and rc's
   owned-return consumption of `return new Shared(...)`. Expected: no extra and no missing
   reference. Task 2's `own_shared_count` asserts `Shared.count` after `make()` and `keepOne()`.
3. **A payload that is not a class:** `Shared<string>` and `Shared<number>`. Expected: the payload
   is released by its own routine (a string), or by nothing (a number). Task 1's `own_shared_basic`
   holds a string.
4. **A handle reassigned to the block it already holds** (`b = a` when `b` is `a`). Expected: the
   count goes up before it goes down, so the block is not freed in between. Task 2's
   `own_shared_count` does it and asserts the count.
5. **A handle created and dropped many times in a loop.** Expected: the count returns to where it
   was and memory stays flat. Task 2's `own_shared_count` does it 1000 times, and Task 5 measures
   200,000.

---

## File map

| File | Change | Task |
| --- | --- | --- |
| `include/TypeScript/TypeScriptTypes.td` | `TypeScript_Shared` type | 1 |
| `include/TypeScript/TypeScriptOps.td` | `SharedNew`, `SharedValueRef`, `SharedCount` | 1 |
| `lib/TypeScript/MLIRGenTypes.cpp` | `Shared` as an embedded name, `Shared<T>` | 1 |
| `lib/TypeScript/MLIRGenExpressions.cpp` | `new Shared(x)`, `Shared.count(s)` | 1, 2 |
| `lib/TypeScript/MLIRGenAccessCall.cpp` | `s.value` | 1 |
| `include/TypeScript/MLIRLogic/MLIRTypeCore.h` | `isNullableTypeNoUnion`, `isNullableOrOptionalType` | 1 |
| `include/TypeScript/MLIRLogic/MLIRTypeIterator.h` | recurse into `Shared<T>` | 1 |
| `include/TypeScript/MLIRLogic/MLIRPrinter.h` | print `Shared<T>` | 1 |
| `lib/TypeScript/LowerToLLVM.cpp` | converter, three lowerings, retain gates, descriptor retain | 1, 3, 4 |
| `lib/TypeScript/LowerToAffineLoops.cpp` | legal ops | 1 |
| `include/TypeScript/MLIRLogic/MLIRTypeHelper.h` | `ownsHeapMemory`, `isSharedHandleType` | 2 |
| `include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h` | counted routines for a handle | 2 |
| `lib/TypeScript/MLIRGenImpl.h` | `markFreshBlockOwned`, `isOwningSlot` for the payload | 2 |
| `lib/TypeScript/OwnershipFacts.h` | `isHandle`, `isPlace` | 3 |
| `lib/TypeScript/OwnershipInferencePass.cpp` | step aside; reads through a handle | 3 |
| `lib/TypeScript/OwnershipSignaturePass.cpp` | no facts on handles; drops | 3 |
| `include/TypeScript/MLIRLogic/TypeOfOpHelper.h` | a `typeof` name, so it gets a descriptor | 4 |
| `lib/TypeScript/MLIRGenCast.cpp` | `any` to `Shared<T>` | 4 |
| `test/tester/own/own_shared_*.ts`, `own_err_shared_*.ts` | tests | 1-4 |
| `test/tester/CMakeLists.txt` | registration | 1-4 |
| spec §23 | amendments (Task 1), results §23.7 (Task 5) | 1, 5 |

All paths below are relative to `tslang/`. Line numbers were read on 8b9d6316; find the code by the
names given, because lines move.

---

### Task 1: the type, the ops, and MLIRGen (no counting yet)

**Files:**
- Modify: `include/TypeScript/TypeScriptTypes.td`, `include/TypeScript/TypeScriptOps.td`
- Modify: `lib/TypeScript/MLIRGenTypes.cpp`, `lib/TypeScript/MLIRGenExpressions.cpp`,
  `lib/TypeScript/MLIRGenAccessCall.cpp`
- Modify: `include/TypeScript/MLIRLogic/MLIRTypeCore.h`, `MLIRTypeIterator.h`, `MLIRPrinter.h`
- Modify: `lib/TypeScript/LowerToLLVM.cpp`, `lib/TypeScript/LowerToAffineLoops.cpp`
- Create: `test/tester/own/own_shared_basic.ts`
- Modify: `test/tester/CMakeLists.txt`
- Modify: the spec, §23.1 and §23.6 (Step 9)

**Interfaces:**
- Produces:
  - `mlir_ts::SharedType` with `getElementType()`, built by `mlir_ts::SharedType::get(elementType)`.
  - `mlir_ts::SharedNewOp::create(builder, loc, SharedType, mlir::Value value)` with `getValue()`.
  - `mlir_ts::SharedValueRefOp(loc, mlir_ts::RefType resultType, mlir::Value shared)` with
    `getShared()` and `getResult()`.
  - `mlir_ts::SharedCountOp(loc, NumberType, mlir::Value shared)`.
  - The CMake variable `shared_models` and the foreach that registers `own_shared_*` tests per model.

- [ ] **Step 1: Write the test.** `test/tester/own/own_shared_basic.ts`:

```ts
// Shared<T> (spec 23): two handles to one block, `.value` read and write, `===` on handles, and a
// payload that is a string. Each handle dropped is followed by reusing freed memory before the
// other is read.
class Node {
    v = 0;
    name = "";
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function main() {
    let a = new Shared(new Node());
    a.value.name = "first" + 1;
    const b = a;
    assert(a === b, "two handles to one block are equal");
    a.value.v = 5;
    assert(b.value.v == 5, "a write through one handle is seen through the other");

    const c = new Shared(new Node());
    assert(a !== c, "two blocks are not equal");

    a = c;
    assert(churn() == 1000);
    assert(b.value.v == 5 && b.value.name == "first1", "the other handle keeps the block");

    const d = b;
    d.value = new Node();
    d.value.v = 7;
    assert(b.value.v == 7, "the value replaced through one handle is seen through the other");

    const t = new Shared("text" + 2);
    const u = t;
    assert(u.value == "text2");
    t.value = "other" + 3;
    assert(churn() == 1000);
    assert(u.value == "other3");

    print("done.");
}
```

- [ ] **Step 2: Register it under gc and none, and watch it fail.**
  - In `test/tester/CMakeLists.txt`, after the `strings_mm` foreach (search
    `foreach(strings_mm rc none gc)`), add:

```cmake
# Shared<T> (spec 23): counted handles. These are not in the own_test list, whose
# test-own-no-counting check forbids __tslang_inc_ref: a handle is counted under own by design.
set(shared_models gc none)
foreach(shared_mm ${shared_models})
    foreach(shared_test own_shared_basic)
        tslang_add_test(NAME test-jit-${shared_mm}-${shared_test} COMMAND test-runner -jit -mm=${shared_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${shared_test}.ts")
        tslang_add_test(NAME test-compile-${shared_mm}-${shared_test} COMMAND test-runner -mm=${shared_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${shared_test}.ts")
    endforeach()
endforeach()
```

  - Build, then run `ctest -C Release -R own_shared -j 8 --timeout 300`.
  - Expected: 4 tests, all FAIL, with "can't resolve name" or similar for `Shared`.

- [ ] **Step 3: The type.** In `include/TypeScript/TypeScriptTypes.td`, after `def TypeScript_Ref`
  (the block that ends with its `builders`), add:

```tablegen
// Shared<T> (own spec 23): a counted handle to a block whose payload is one T.
def TypeScript_Shared : TypeScript_Type<"Shared"> {
  let mnemonic = "shared";

  let description = [{
    Syntax:

    ```
    shared-type ::= `shared` `<` type `>`
    ```

    Examples:

    ```mlir
    shared<!ts.class<Node>>
    ```
  }];
  let parameters = (ins "Type":$elementType);

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>
  ];
}
```

- [ ] **Step 4: The ops.** In `include/TypeScript/TypeScriptOps.td`, after `def TypeScript_StringCopyOp`,
  add:

```tablegen
// Shared<T> (own spec 23). `new Shared(x)`: a block of one T, which `value` moves into.
def TypeScript_SharedNewOp : TypeScript_Op<"SharedNew"> {
  let arguments = (ins AnyType:$value);
  let results = (outs Res<TypeScript_Shared, "", [MemAlloc]>:$result);
}

// The place `s.value`: a reference to the handle's payload. It is read with ts.Load and written
// with ts.Store, as a field is.
def TypeScript_SharedValueRefOp : TypeScript_Op<"SharedValueRef"> {
  let arguments = (ins TypeScript_Shared:$shared);
  let results = (outs TypeScript_Ref:$result);
}

// `Shared.count(s)`: the number of handles under rc and own, -1 under gc and none.
def TypeScript_SharedCountOp : TypeScript_Op<"SharedCount"> {
  let arguments = (ins TypeScript_Shared:$shared);
  let results = (outs TypeScript_Number:$result);
}
```

  Build with `--parallel 3`. Expected: it compiles. If `TypeScript_Shared` cannot be used as an
  operand constraint, wrap it the way `TypeScript_AnyClassOrValueRef` is
  (`TypeScriptTypes.td`, `AnyTypeOf<[...]>`): `def TypeScript_SharedLike : AnyTypeOf<[TypeScript_Shared]>;`

- [ ] **Step 5: MLIRGen.** Each change is small.
  - **The embedded name** (`lib/TypeScript/MLIRGenTypes.cpp`):
    - Add `{"Shared", true}` to the static maps in `isEmbededTypeWithBuiltins` and
      `isEmbededTypeWithNoBuiltins`, next to `{"BoxedObject", true}`.
    - In both `getEmbeddedTypeWithParamBuiltins` and `getEmbeddedTypeWithParamNoBuiltins`, add
      `Shared` to the `EmbeddedType` enum, add `.Case("Shared", EmbeddedType::Shared)` to the
      `StringSwitch`, and add this case to the `switch (kind)`:

```cpp
            case EmbeddedType::Shared:
                return mlir_ts::SharedType::get(type);
```

  - **`new Shared(x)`** (`lib/TypeScript/MLIRGenExpressions.cpp`, `mlirGen(NewExpression ...)`):
    in the `if (result.failed())` / `typeExpression == SyntaxKind::Identifier` fallback, before
    `findEmbeddedType`, add:

```cpp
                if (name == "Shared")
                {
                    return mlirGenNewShared(location, newExpression, genContext);
                }
```

    Then add the method to `MLIRGenImpl` (declare it in `MLIRGenImpl.h` beside the other
    `mlirGen(NewExpression ...)` helpers; define it in `MLIRGenExpressions.cpp`):

```cpp
    // `new Shared(x)` (own spec 23): T is the type argument, else the receiver's, else x's.
    ValueOrLogicalResult MLIRGenImpl::mlirGenNewShared(mlir::Location location, NewExpression newExpression,
                                                       const GenContext &genContext)
    {
        if (!newExpression->arguments || newExpression->arguments->size() != 1)
        {
            emitError(location, "new Shared(value) takes one argument");
            return mlir::failure();
        }

        auto result = mlirGen(newExpression->arguments->front(), genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto value = V(result);

        mlir::Type elementType;
        if (newExpression->typeArguments && newExpression->typeArguments->size() == 1)
        {
            elementType = getType(newExpression->typeArguments->front(), genContext);
        }
        else if (auto receiver = dyn_cast_or_null<mlir_ts::SharedType>(genContext.receiverType))
        {
            elementType = receiver.getElementType();
        }
        else
        {
            elementType = mth.wideStorageType(value.getType());
        }

        if (!elementType)
        {
            return mlir::failure();
        }

        CAST_A(stored, location, elementType, value, genContext);
        return V(builder.create<mlir_ts::SharedNewOp>(location, mlir_ts::SharedType::get(elementType), stored));
    }
```

    Use the casting macro this file already uses for "cast a value to a type" (search
    `CAST_A(` in `MLIRGenExpressions.cpp`). If the receiver-type field has another name, use the
    one `NewClassInstance` reads (search `receiverType` in `mlirGen(NewExpression`).
  - **`s.value`** (`lib/TypeScript/MLIRGenAccessCall.cpp`, the `mlir::TypeSwitch` over
    `actualType` in `mlirGenPropertyAccessExpressionBaseLogic`): add a case beside
    `.Case<mlir_ts::RefType>`:

```cpp
            .Case<mlir_ts::SharedType>([&](auto sharedType) -> mlir::Value {
                if (name != "value")
                {
                    emitError(location, "Shared<T> has no member '") << name << "'";
                    return mlir::Value();
                }

                auto elementType = sharedType.getElementType();
                auto valueRef = builder.create<mlir_ts::SharedValueRefOp>(
                    location, mlir_ts::RefType::get(elementType), objectValue);
                return builder.create<mlir_ts::LoadOp>(location, elementType, valueRef);
            })
```

    Use the local names that function already uses for the member name and the object value.
    `s.value = x` then needs no change: `mlirGenSaveLogicOneItem` (`MLIRGenImpl.h`) finds the
    `LoadOp` and stores through its reference, the `SharedValueRef`.
  - **`Shared.count(s)`** (`mlirGen(CallExpression ...)` in `MLIRGenExpressions.cpp`): before the
    general call path, add:

```cpp
        // `Shared.count(s)` (own spec 23.1)
        if (auto access = callExpression->expression.as<PropertyAccessExpression>();
            callExpression->expression == SyntaxKind::PropertyAccessExpression &&
            access->expression == SyntaxKind::Identifier &&
            MLIRHelper::getName(access->expression.as<Identifier>()) == "Shared" &&
            MLIRHelper::getName(access->name) == "count" && !resolve("Shared", genContext))
        {
            if (!callExpression->arguments || callExpression->arguments->size() != 1)
            {
                emitError(location, "Shared.count(handle) takes one argument");
                return mlir::failure();
            }

            auto result = mlirGen(callExpression->arguments->front(), genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto handle = V(result);
            if (!isa<mlir_ts::SharedType>(handle.getType()))
            {
                emitError(location, "Shared.count takes a Shared<T>");
                return mlir::failure();
            }

            return V(builder.create<mlir_ts::SharedCountOp>(location, getNumberType(), handle));
        }
```

    `!resolve("Shared", genContext)` stands for "no program symbol named `Shared`". Use whatever
    lookup the new-expression fallback relies on to find that `mlirGen(identifier)` failed (for
    example, a `findDeclaration`/`resolveIdentifier` call; search how `mlirGen(location, name)`
    decides "can't resolve name").
- [ ] **Step 6: Type helpers.**
  - **`include/TypeScript/MLIRLogic/MLIRTypeCore.h`:** add `mlir_ts::SharedType` to the lists in
    `isNullableTypeNoUnion` and `isNullableOrOptionalType`, beside `ClassType`. This makes `===`
    compare pointers and `null` compare and assign.
  - **`include/TypeScript/MLIRLogic/MLIRTypeIterator.h`:** add a case beside the `RefType` one that
    visits `getElementType()`, so a generic `T` inside `Shared<T>` is found.
  - **`include/TypeScript/MLIRLogic/MLIRPrinter.h`, `printType`:** add, beside the `RefType` case:

```cpp
            .template Case<mlir_ts::SharedType>([&](auto t) {
                out << "Shared<";
                printType(out, t.getElementType());
                out << ">";
            })
```

- [ ] **Step 7: Lowering.**
  - **`lib/TypeScript/LowerToLLVM.cpp`, `populateTypeScriptConversionPatterns`:** add beside the
    `RefType` conversion:

```cpp
    converter.addConversion([&](mlir_ts::SharedType type) {
        return LLVM::LLVMPointerType::get(m.getContext());
    });
```

  - **The lowerings**, beside `StringCopyOpLowering`:

```cpp
// `new Shared(x)` (own spec 23): a heap block with the header every block has (a count under rc
// and own, born 0 like any block), whose payload is the value.
struct SharedNewOpLowering : public TsLlvmPattern<mlir_ts::SharedNewOp>
{
    using TsLlvmPattern<mlir_ts::SharedNewOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SharedNewOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        auto elementType = cast<mlir_ts::SharedType>(op.getType()).getElementType();
        auto handle = ch.MemoryAlloc(elementType, MemoryAllocSet::Zero);
        rewriter.create<LLVM::StoreOp>(op->getLoc(), transformed.getValue(), handle);
        rewriter.replaceOp(op, ValueRange{handle});
        return success();
    }
};

// `s.value`: the payload starts at the handle.
struct SharedValueRefOpLowering : public TsLlvmPattern<mlir_ts::SharedValueRefOp>
{
    using TsLlvmPattern<mlir_ts::SharedValueRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SharedValueRefOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        rewriter.replaceOp(op, ValueRange{transformed.getShared()});
        return success();
    }
};

// `Shared.count(s)`: the header word under rc and own; gc and none never write it.
struct SharedCountOpLowering : public TsLlvmPattern<mlir_ts::SharedCountOp>
{
    using TsLlvmPattern<mlir_ts::SharedCountOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SharedCountOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = op->getLoc();
        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        auto f64Type = rewriter.getF64Type();
        if (!tsLlvmContext->compileOptions.tracksOwnership())
        {
            rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, f64Type, rewriter.getF64FloatAttr(-1.0));
            return success();
        }

        auto llvmIndexType = ch.getIndexType();
        auto blockPtr = ch.getBlockPtrFromPayloadPtr(loc, transformed.getShared(), llvmIndexType);
        auto count = rewriter.create<LLVM::LoadOp>(loc, llvmIndexType, blockPtr);
        rewriter.replaceOpWithNewOp<LLVM::SIToFPOp>(op, f64Type, count);
        return success();
    }
};
```

    Use the pattern base class, adaptor and helper names that `StringCopyOpLowering` uses in this
    file, and the index-type getter `_MemoryAlloc` uses (`LLVMCodeHelperBase.h`). If `number`
    lowers to another type than `f64`, use `getTypeConverter()->convertType(op.getType())`.
  - Add the three patterns to the `patterns.insert<...>` list beside `StringCopyOpLowering`.
  - **`lib/TypeScript/LowerToAffineLoops.cpp`, `AddTsAffineLegalOps`:** add
    `mlir_ts::SharedNewOp, mlir_ts::SharedValueRefOp, mlir_ts::SharedCountOp` to the
    `addLegalOp<...>` list beside `mlir_ts::StringCopyOp`.
- [ ] **Step 8: Run the test.**
  - Build, then run `ctest -C Release -R own_shared -j 8 --timeout 300`. Expected: 4/4 pass.
  - Also run `test-runner -mm=rc <file>` once by hand. It should pass too, but it leaks: rc does
    not count handles until Task 2.
  - Check the cross-module type: write a scratch module exporting
    `export function make(): Shared<number> { return new Shared(1); }`, compile it with
    `--emit=mlir` and `--export`, the way the `export_*` corpus files are, and confirm that
    `__decls` prints `Shared<number>`.
- [ ] **Step 9: The spec.** In §23.1:
  - Replace the bullets for `s.value` and `s.value = x` with one bullet: "**`s.value`** is a
    `ts.Load` of the place `ts.SharedValueRef(s)`, and **`s.value = x`** stores into that place
    the way a field store does (MLIRGen's retain, `ts.ReleaseSlot` and `ts.Store`), so `x` moves in
    and the old `T` is released. One op instead of two lets the place-read and store machinery of
    MLIRGen and own apply unchanged."
  - In the decisions table and §23.6, rename the graph test's content: "`own_shared_graph`: a DAG
    (one item under two parents) and a singly linked list walked with `cur = cur.value.next`". Move
    "a doubly linked list and a node with a parent link" to `own_shared_cycle`, since both are
    cycles and leak (§23.4).
  - Keep CRLF.
- [ ] **Step 10: Commit.**

```bash
git add include/TypeScript lib/TypeScript test/tester/own/own_shared_basic.ts test/tester/CMakeLists.txt docs/superpowers/specs/2026-09-24-own-memory-model-design.md
git commit -m "-mm=own Shared<T>: the type, its ops and MLIRGen (spec 23.1)"
```

---

### Task 2: counting a handle (rc)

**Files:**
- Modify: `include/TypeScript/MLIRLogic/MLIRTypeHelper.h`
- Modify: `include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h`
- Modify: `lib/TypeScript/MLIRGenImpl.h`, `lib/TypeScript/MLIRGenExpressions.cpp`
- Modify: `lib/TypeScript/LowerToLLVM.cpp` (the GC root list)
- Create: `test/tester/own/own_shared_count.ts`, `test/tester/own/own_shared_graph.ts`
- Modify: `test/tester/CMakeLists.txt`

**Interfaces:**
- Consumes: `SharedType`, `SharedNewOp`, `SharedValueRefOp`, `SharedCountOp`, `shared_models` (Task 1).
- Produces: `static bool MLIRTypeHelper::isSharedHandleType(mlir::Type type)`. It is true for a
  `SharedType`, or an `OptionalType` of one, or a `UnionType` whose members are one `SharedType` and
  `NullType`/`UndefinedType` only. Tasks 3 and 4 use it.

- [ ] **Step 1: Write the tests.** `test/tester/own/own_shared_count.ts` (rc and own only, since
  gc and none keep no count):

```ts
// Shared<T> (spec 23.1): Shared.count after each copy and drop - a handle made, copied, given to a
// function that keeps it, returned from one, reassigned to the block it holds, and copied in a loop.
class Node {
    v = 0;
}

function keepOne(s: Shared<Node>, into: Shared<Node>[]) {
    into.push(s);
}

function dropLast(list: Shared<Node>[]) {
    list.pop();
}

function make() {
    return new Shared(new Node());
}

function main() {
    const a = new Shared(new Node());
    assert(Shared.count(a) == 1, "one handle");

    let b = a;
    assert(Shared.count(a) == 2, "copied");

    b = a;
    assert(Shared.count(a) == 2, "reassigned to the block it holds");

    const list: Shared<Node>[] = [];
    keepOne(a, list);
    assert(Shared.count(a) == 3, "kept by a callee");

    b = new Shared(new Node());
    assert(Shared.count(a) == 2 && Shared.count(b) == 1, "one handle moved to another block");

    dropLast(list);
    assert(Shared.count(a) == 1, "the kept one dropped");

    const m = make();
    assert(Shared.count(m) == 1, "returned from a function");

    for (let i = 0; i < 1000; i++) {
        const t = a;
        assert(Shared.count(t) == 2, "copied in a loop");
    }

    assert(Shared.count(a) == 1, "every copy in the loop dropped");

    print("done.");
}
```

  `test/tester/own/own_shared_graph.ts`:

```ts
// Shared<T> (spec 23): a DAG - one item under two parents - and a singly linked list walked with
// `cur = cur.value.next`. Handles are dropped and their memory reused before the rest is read.
class Item {
    v = 0;
    name = "";
    next: Shared<Item> | null = null;
}

class Parent {
    child: Shared<Item> | null = null;
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function list(n: number) {
    let head: Shared<Item> | null = null;
    for (let i = 0; i < n; i++) {
        const item = new Shared(new Item());
        item.value.v = i;
        item.value.next = head;
        head = item;
    }

    return head;
}

function main() {
    const child = new Shared(new Item());
    child.value.name = "child" + 1;
    let p1 = new Parent();
    const p2 = new Parent();
    p1.child = child;
    p2.child = child;
    p1 = new Parent();
    assert(churn() == 1000);
    const c = p2.child;
    assert(c !== null && c.value.name == "child1", "the other parent keeps the child");

    let sum = 0;
    let cur = list(10);
    while (cur) {
        sum += cur.value.v;
        cur = cur.value.next;
    }

    assert(sum == 45, "the list walked to its end");
    assert(cur === null);

    print("done.");
}
```

- [ ] **Step 2: Register them and watch them fail.** In `CMakeLists.txt`:
  - change `set(shared_models gc none)` to `set(shared_models gc none rc)`;
  - add `own_shared_graph` to the inner `foreach(shared_test ...)` list;
  - add after that foreach:

```cmake
# Shared.count is a count only under rc and own.
foreach(shared_count_mm rc)
    tslang_add_test(NAME test-jit-${shared_count_mm}-own_shared_count COMMAND test-runner -jit -mm=${shared_count_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_shared_count.ts")
    tslang_add_test(NAME test-compile-${shared_count_mm}-own_shared_count COMMAND test-runner -mm=${shared_count_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_shared_count.ts")
endforeach()
```

  Build, then run `ctest -C Release -R own_shared -j 8 --timeout 300`. Expected:
  - `own_shared_count` fails under rc (the count stays 0: nothing counts a handle yet);
  - `own_shared_basic` and `own_shared_graph` may pass under rc, because a leak is invisible;
  - gc and none pass.

  If `own_shared_graph` fails to compile under gc or none (a `Shared<Item> | null` narrowing),
  that is Review Focus 1: fix it in this task. The union must be tag-free and narrow like a class
  union; check `isUnionTypeNeedsTag` and the narrowing helpers.
- [ ] **Step 3: The type predicates** (`include/TypeScript/MLIRLogic/MLIRTypeHelper.h`):
  - In `ownsHeapMemory(location, type, visiting)`, add `isa<mlir_ts::SharedType>(type)` to the first
    "owns its own block" test, beside `ClassType`. Do not add `SharedType` to
    `getOwnershipFieldTypes`: the block's payload is released by the handle's routine.
  - Add:

```cpp
    // A handle (own spec 23.2): Shared<T>, or a tag-free union or an optional of one with null or
    // undefined - one pointer whose block is counted in every model that tracks ownership.
    static bool isSharedHandleType(mlir::Type type)
    {
        if (isa<mlir_ts::SharedType>(type))
        {
            return true;
        }

        if (auto optionalType = dyn_cast<mlir_ts::OptionalType>(type))
        {
            return isSharedHandleType(optionalType.getElementType());
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            mlir::Type shared;
            for (auto member : unionType.getTypes())
            {
                if (isa<mlir_ts::NullType, mlir_ts::UndefinedType>(member))
                {
                    continue;
                }

                if (!isa<mlir_ts::SharedType>(member) || (shared && shared != member))
                {
                    return false;
                }

                shared = member;
            }

            return !!shared;
        }

        return false;
    }
```

- [ ] **Step 4: The routines** (`include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h`):
  - In `buildRetainBody`, add `isa<mlir_ts::SharedType>(type)` to the "reference to a block of its
    own" condition beside `ClassType`, so it is `emitIncRef` of the loaded pointer.
  - In `buildBody`, after the class/object branch, add:

```cpp
        // a handle (own spec 23.2) is counted in every model that tracks ownership: the last one
        // releases the payload and frees the block
        if (auto sharedType = dyn_cast<mlir_ts::SharedType>(type))
        {
            auto handle = rewriter.create<LLVM::LoadOp>(loc, ptrTy, slotPtr);
            emitIfLastReference(handle, [&]() {
                releaseSlot(sharedType.getElementType(), handle);
                emitFreeBlock(handle);
            }, /*counted=*/true);
            return;
        }
```

  - Give `emitIfLastReference` a `bool counted = false` parameter, and choose the test with it:

```cpp
        auto wasLast = compileOptions.memoryModel == MemoryModelOwn && !counted ? emitIsMortal(payloadPtr)
                                                                               : emitDecRef(payloadPtr);
```

    `releaseSlot(type, slotPtr)` is the helper that calls a type's release routine on a slot. The
    handle points at the payload, which is the `T`'s slot. If the helper has another name, use the
    one `releaseFields` calls per field.
  - **A tag-free union `Shared<T> | null`** is released and retained through the union branches.
    Check with `--emit=llvm -mm=rc` on `own_shared_graph.ts` that the `next` field's release calls
    the handle's `tsrel_` routine, not a tag dispatch. If it dispatches on a tag the union does not
    have, make the tag-free union of one `SharedType` use the `SharedType` routine (the same way a
    tag-free `C | null` uses the class routine; find that branch and add `SharedType` beside
    `ClassType`).
- [ ] **Step 5: MLIRGen's ownership ops for a handle** (`lib/TypeScript/MLIRGenImpl.h`):
  - In `markFreshBlockOwned`, widen the test to
    `isa<mlir_ts::StringType, mlir_ts::AnyType, mlir_ts::SharedType>(value.getType())`. A new
    handle then carries its birth reference the way a new string does (§9.25 of the rc document).
  - In `mlirGenNewShared` (Task 1), give the value to the block, then mark the new handle:

```cpp
        CAST_A(stored, location, elementType, value, genContext);
        // the block owns what it holds, as an array literal owns its elements
        mlirGenRetainCaptured(location, mlir::ValueRange{stored});
        auto handle = builder.create<mlir_ts::SharedNewOp>(location, mlir_ts::SharedType::get(elementType), stored);
        markFreshBlockOwned(location, handle);
        return V(handle);
```

  - **`isOwningSlot`:** add an `isOwnedSharedValueSlot` beside `isOwnedFieldSlot`, and include it
    in `isOwningSlot`'s `||` chain:

```cpp
    // `s.value` (own spec 23): the payload is owned by the handle's block, as a field by its object
    bool isOwnedSharedValueSlot(mlir::Location location, mlir::Value reference)
    {
        auto valueRef = reference.getDefiningOp<mlir_ts::SharedValueRefOp>();
        return valueRef &&
               mth.ownsHeapMemory(location, cast<mlir_ts::RefType>(reference.getType()).getElementType());
    }
```

  - **`LowerToLLVM.cpp`:** in the GC root type test (the list that names `ClassType` near the
    `GC_ENABLE` code in `VariableOpLowering`), add `SharedType` beside `ClassType`.
- [ ] **Step 6: Run the tests.**
  - Build, then run `ctest -C Release -R own_shared -j 8 --timeout 300`. Expected: all pass (gc,
    none and rc, plus `own_shared_count` under rc).
  - Run `test-runner -noopt -mm=rc` and `test-runner -noopt -jit -mm=rc` on all three files.
    Expected: they pass.
  - Run `ctest -C Release -R "own|rc" -j 8 --timeout 300`. Expected: no new failure (nothing
    existing uses `Shared`).
- [ ] **Step 7: Commit.**

```bash
git add include/TypeScript lib/TypeScript test/tester/own/own_shared_count.ts test/tester/own/own_shared_graph.ts test/tester/CMakeLists.txt
git commit -m "-mm=own Shared<T>: a handle is counted, under rc first (spec 23.2)"
```

---

### Task 3: own steps aside, and reads through a handle

**Files:**
- Modify: `lib/TypeScript/OwnershipFacts.h`, `lib/TypeScript/OwnershipInferencePass.cpp`,
  `lib/TypeScript/OwnershipSignaturePass.cpp`
- Modify: `lib/TypeScript/LowerToLLVM.cpp` (the retain gates)
- Create: `test/tester/own/own_err_shared_borrow_write.ts`, `own_err_shared_borrow_call.ts`,
  `own_err_shared_borrow_push.ts`, `own_err_shared_moved_in.ts`
- Modify: `test/tester/CMakeLists.txt`

**Interfaces:**
- Consumes: `MLIRTypeHelper::isSharedHandleType` (Task 2), `SharedValueRefOp`, `SharedNewOp` (Task 1).
- Produces:
  - `inline bool isHandle(mlir::Value value)` in `OwnershipFacts.h`;
  - `inline bool touchesHandle(mlir::Operation *op)`, true for a `Retain`/`Release` of a handle
    value, or a `RetainSlot`/`ReleaseSlot` of a slot holding one.

- [ ] **Step 1: Write the negative tests.**

  `own_err_shared_borrow_write.ts`:

```ts
// -mm=own rejects (spec 23.3): a borrow read through one handle, used after a write through another
// handle that may reach the same block.
class Node {
    v = 0;
}

function main() {
    const a = new Shared(new Node());
    const b = a;
    const n = a.value;
    b.value = new Node();
    print(n.v);
}
```

  `own_err_shared_borrow_call.ts`:

```ts
// -mm=own rejects (spec 23.3): a borrow read through a handle, used after a call that may drop -
// `touch` writes through a handle it reaches from a global.
class Node {
    v = 0;
}

let kept: Shared<Node> | null = null;

function touch() {
    const k = kept;
    if (k) k.value = new Node();
}

function main() {
    const a = new Shared(new Node());
    kept = a;
    const n = a.value;
    touch();
    print(n.v);
}
```

  `own_err_shared_borrow_push.ts`:

```ts
// -mm=own rejects (spec 23.3): an element borrowed through one handle, used after a push through
// another, which may move the elements.
class Node {
    v = 0;
}

class Bag {
    items: Node[] = [];
}

function main() {
    const a = new Shared(new Bag());
    const b = a;
    a.value.items.push(new Node());
    const first = a.value.items[0];
    b.value.items.push(new Node());
    print(first.v);
}
```

  `own_err_shared_moved_in.ts`:

```ts
// -mm=own rejects: a class value given to `new Shared(x)` moves into the block, so it cannot be used
// after (spec 23.1, 2.3).
class Node {
    v = 0;
}

function main() {
    const n = new Node();
    const s = new Shared(n);
    print(n.v, s.value.v);
}
```

- [ ] **Step 2: Register them; add own to the positives; watch them fail.** In `CMakeLists.txt`:
  - add to `own_error_cases`:

```cmake
    "own_err_shared_borrow_write|but is used here after it may be released or overwritten"
    "own_err_shared_borrow_call|but is used here after it may be released or overwritten"
    "own_err_shared_borrow_push|but is used here after it may be released or overwritten"
    "own_err_shared_moved_in|is used here after its value was moved"
```

  - change `set(shared_models gc none rc)` to `set(shared_models gc none rc own)`, and
    `foreach(shared_count_mm rc)` to `foreach(shared_count_mm rc own)`;
  - add the verifier over every positive, under own and rc:

```cmake
foreach(shared_verify_mm own rc)
    foreach(shared_test own_shared_basic own_shared_graph own_shared_count)
        add_test(NAME test-${shared_verify_mm}-verify-ownership-${shared_test}
                 COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=${shared_verify_mm} --verify-ownership
                         "${PROJECT_SOURCE_DIR}/test/tester/own/${shared_test}.ts")
        set_tests_properties(test-${shared_verify_mm}-verify-ownership-${shared_test}
                             PROPERTIES FAIL_REGULAR_EXPRESSION "error|Stack dump")
    endforeach()
endforeach()
```

  Build, then run `ctest -C Release -R "own_shared|own-err-own_err_shared|verify-ownership-own_shared" -j 8 --timeout 300`.
  Expected:
  - the own positives fail, with "takes a second reference" or "left a retain behind";
  - `own_err_shared_moved_in` passes (a taker already moves), and the other three negatives give
    their own first errors or compile;
  - the rest pass.

  Record each negative's actual first message now.
- [ ] **Step 3: The facts** (`lib/TypeScript/OwnershipFacts.h`):

```cpp
// A handle (spec 23.2): counted in every model that tracks ownership, and left alone by own.
inline bool isHandle(mlir::Value value)
{
    return value && mlir_ts::MLIRTypeHelper::isSharedHandleType(value.getType());
}

inline bool touchesHandle(mlir::Operation *op)
{
    if (mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp>(op))
    {
        return isHandle(op->getOperand(0));
    }

    if (mlir::isa<mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp>(op))
    {
        auto slotType = mlir::cast<mlir_ts::RefType>(op->getOperand(0).getType());
        return mlir_ts::MLIRTypeHelper::isSharedHandleType(slotType.getElementType());
    }

    return false;
}
```

  Use the namespace `MLIRTypeHelper` is in, as the file's other uses show. In `isPlace(ref)`, add
  `SharedValueRefOp` beside `PropertyRefOp`/`ElementRefOp`: `s.value` is a place.
- [ ] **Step 4: The inference pass steps aside** (`lib/TypeScript/OwnershipInferencePass.cpp`). The
  goal is "never erased, never reported" for a handle. The safety net comes first:
  - **Never erased.** Immediately before the loop that erases `toErase` in `analyze`, add:

```cpp
        // a handle's retains and releases are rc's, and stay (spec 23.2)
        toErase.remove_if([](mlir::Operation *op) { return touchesHandle(op); });
```

  - **Never decided.** In the walk that collects `retains` and `candidates`, skip ops for which
    `touchesHandle(op)` holds: do not push them into `retains`, and do not add their value to
    `candidates`, `drops` or `slotLoadOf` lists.
  - In the top-level retain loop, add `if (touchesHandle(op)) { continue; }` as the first statement.
  - For a call's kept arguments (the `keepingCalls` loop that calls `reportGivenNotOwned`), skip an
    argument with `isHandle(argument)`.
  - In `decideClosures`/`decideEscaping`, do not erase or report a capture retain whose value is a
    handle (`touchesHandle`).
  - **A handle read out of a place is a counted copy, not a borrow (§23.3).** In `placeReadOf`,
    return null when the loaded value `isHandle`.
  - **A read through a handle (§23.3).** `s.value` is now a place (`isPlace`), so `Load(SharedValueRef)`
    is a place read, and `chainOf` walks through it. Make `chainOf` treat a `SharedValueRefOp` link
    as a place whose root is the handle, with root kind `NotOwned`. A handle is reachable through
    aliases, so it is never "owned". Then:
    - `dropsChain`'s call rule already ends a borrow with a not-owned root at any call that is not
      `callNoDrops`;
    - a `ReleaseSlot` of a `SharedValueRef` place (a store `x.value = …` through any handle) must
      match a chain that has a `SharedValueRef` place of the same element type. Add that beside
      the position-and-type match `dropsChain` does for a `PropertyRef`;
    - a store into a field of a `T` through any handle (`x.value.f = …`) is a `ReleaseSlot` of a
      `PropertyRef` of the same position and type, which `dropsChain` already matches by type;
    - add `ArrayPushOp`, and `ArrayUnshiftOp` if it exists, to the ops the walk collects into
      `drops` beside `ArrayPopOp`, but only when the array operand's `chainOf` reaches a
      `SharedValueRef`. It is a drop for an element place of the same element type, as `pop` is.
  - `describePlace` should name a `SharedValueRef` place `'<handle>.value'` (e.g. `'a.value'`).
- [ ] **Step 5: The signature pass** (`lib/TypeScript/OwnershipSignaturePass.cpp`):
  - **No facts on handles:** in `setFacts`, leave out of `__own_params` any parameter whose type
    `isSharedHandleType`, and do not set `__own_result_borrows` for a result that is a handle.
  - **`keeps()`:** a `SharedNewOp` whose operand is the parameter keeps it, like a store into a
    place.
  - **`dropsInBody`:**
    - a `ReleaseOp` of a handle value, or a `ReleaseSlotOp` of a slot holding one, is a drop. It may
      be the last handle, and its payload's release can free anything the payload owns;
    - a `ReleaseSlotOp` of a `SharedValueRef` place is a drop. In `reachesOutside`, treat a
      `SharedValueRef` as reaching outside: any handle may be shared.
- [ ] **Step 6: The lowering gates** (`lib/TypeScript/LowerToLLVM.cpp`):
  - In `RetainOpLowering` and `RetainSlotOpLowering`, compute
    `auto counted = MLIRTypeHelper::isSharedHandleType(<value type>)`. Use the reference's type,
    and for a slot, the `RefType`'s element type. Then:
    - raise "ownership inference left a retain behind" under own only when `!counted`;
    - emit the retain when `isRefCounted() || (tracksOwnership() && counted)`.
  - `RetainCellOpLowering` is unchanged, since a cell is not a handle.
  - In the two `VariableOpLowering` sites gated on `isRefCounted()` that retain a captured slot's
    initial value with `emitRetainSlot`, use the same `|| (tracksOwnership() && counted)` on the
    slot's element type.
- [ ] **Step 7: Run the tests.**
  - Build, then run
    `ctest -C Release -R "own_shared|own-err-own_err_shared|verify-ownership-own_shared" -j 8 --timeout 300`.
    Expected: all pass. That is every positive under gc, none, rc and own (AOT and JIT), the count
    test under rc and own, the four negatives with their regexes, and the verifier quiet.
  - If a negative's first error differs from its regex, keep a regex that names the §23.3 rule or
    the move, and say so in the commit message.
  - Run `test-runner -noopt -mm=own` and `-noopt -jit -mm=own` on every positive.
  - Run the full `ctest -C Release -R own -j 8 --timeout 300`. Expected: every existing own test
    still passes (490 before this phase).
- [ ] **Step 8: Teeth.** Commit or back up first. Each switch is temporary. Clear
  `test/tester/own/__jit` and the build's `own/__jit` between JIT runs.
  - (a) Remove the `toErase.remove_if(...)` line and the `touchesHandle` skips in the retain loop:
    the own positives must fail (to compile, or at run time).
  - (b) Make `chainOf` treat a `SharedValueRef` root as owned: `own_err_shared_borrow_call` must
    compile.
  - (c) Drop the `SharedValueRef` match from `dropsChain`: `own_err_shared_borrow_write` must
    compile.
  - (d) Drop the push rule: `own_err_shared_borrow_push` must compile. If it errors anyway, the
    existing rules already end that borrow. Record which, for §23.7.
  - Revert all four.
- [ ] **Step 9: Commit.**

```bash
git add lib/TypeScript test/tester/own/own_err_shared_*.ts test/tester/CMakeLists.txt
git commit -m "-mm=own Shared<T>: own leaves handles to the count; reads through a handle (spec 23.2, 23.3)"
```

---

### Task 4: handles in containers, `any`, closures, and cycles

**Files:**
- Modify: `include/TypeScript/MLIRLogic/TypeOfOpHelper.h`, `lib/TypeScript/LowerToLLVM.cpp`
  (`TypeDescriptorOpLowering`), `lib/TypeScript/MLIRGenCast.cpp`, and wherever the test shows a gap
- Create: `test/tester/own/own_shared_containers.ts`, `test/tester/own/own_shared_cycle.ts`
- Modify: `test/tester/CMakeLists.txt`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the tests.** `own_shared_containers.ts`:

```ts
// Shared<T> (spec 23) in containers: one node in two arrays, a `Shared<T> | null` field, a closure
// capturing a handle, a handle boxed into `any` and back, and a handle returned from a function.
// Each owner but one is dropped and its memory reused before the last is read.
class Node {
    v = 0;
    tag = "";
}

class Holder {
    s: Shared<Node> | null = null;
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function make(v: number) {
    const s = new Shared(new Node());
    s.value.v = v;
    s.value.tag = "t" + v;
    return s;
}

function main() {
    const s = make(3);

    let left: Shared<Node>[] = [s];
    const right: Shared<Node>[] = [s];
    left = [];
    assert(churn() == 1000);
    assert(right[0].value.v == 3 && right[0].value.tag == "t3", "the other array keeps the node");

    const h = new Holder();
    h.s = right[0];
    const read = () => s.value.v;
    assert(read() == 3, "a closure reads through its handle");

    const boxed: any = h.s;
    const back = <Shared<Node>>boxed;
    assert(back === s, "a handle boxed into any comes back the same");
    assert(back.value.tag == "t3");

    h.s = null;
    assert(churn() == 1000);
    assert(read() == 3 && back.value.v == 3, "the closure and the unboxed handle keep the node");

    print("done.");
}
```

  `own_shared_cycle.ts`:

```ts
// Shared<T> (spec 23.4): a doubly linked pair and a child with a link to its parent are cycles,
// which are never freed (as under rc); the program still runs.
class Item {
    v = 0;
    next: Shared<Item> | null = null;
    prev: Shared<Item> | null = null;
}

function main() {
    for (let i = 0; i < 100; i++) {
        const a = new Shared(new Item());
        const b = new Shared(new Item());
        a.value.next = b;
        b.value.prev = a;
        a.value.v = i;
        const p = b.value.prev;
        assert(p !== null && p.value.v == i);

        const parent = new Shared(new Item());
        const child = new Shared(new Item());
        parent.value.next = child;
        child.value.prev = parent;
    }

    print("done.");
}
```

- [ ] **Step 2: Register them and watch them fail.** Add `own_shared_containers own_shared_cycle`
  to the inner `foreach(shared_test ...)` list and to the `shared_verify_mm` list. Build, then run
  `ctest -C Release -R own_shared -j 8 --timeout 300`. Expected: `own_shared_containers` fails in
  some models. The `any` box has no type descriptor for a handle yet, and the `<Shared<Node>>`
  cast from `any` is not supported. `own_shared_cycle` may already pass.
- [ ] **Step 3: A descriptor for a handle.**
  - **`TypeOfOpHelper.h`, `typeOfAsString`:** return `"object"` for a `SharedType`. That is what
    JavaScript's `typeof` would say of a wrapper object. The descriptor is keyed by the type's hash,
    so `Shared<A>` and `Shared<B>` still get distinct descriptors.
  - **`LowerToLLVM.cpp`, `TypeDescriptorOpLowering`:** under own, the retain routine name is empty
    for every type. Keep it for a handle:

```cpp
        auto retainRoutineName =
            tsLlvmContext->compileOptions.memoryModel == MemoryModelOwn &&
                    !MLIRTypeHelper::isSharedHandleType(descriptorType)
                ? std::string()
                : orl.getOrCreateRetainRoutine(descriptorType);
```

- [ ] **Step 4: `any` back to `Shared<T>`** (`lib/TypeScript/MLIRGenCast.cpp`). Find where a cast
  from `any` to a class type is generated: it checks the box's type tag and unboxes. Make
  `SharedType` take the same path as `ClassType`. The tag it compares against is the handle's
  descriptor from Step 3.
  - If that path compares the `typeof` name only ("object"), an `any` holding a plain object would
    also pass the check. Record that in §23.7 as a limit.
  - If the class path cannot be reused, stop and report the reason (BLOCKED).
- [ ] **Step 5: Run the tests.**
  - Build, then run `ctest -C Release -R "own_shared|verify-ownership-own_shared" -j 8 --timeout 300`.
    Expected: all pass under gc, none, rc and own, AOT and JIT.
  - Run `-noopt` under own and rc for both files.
  - If own rejects `own_shared_containers` with an error that names a non-handle value, read it:
    - a rule of §23.3 means the test is wrong; rewrite the shape and say so in the commit;
    - anything else is a gap in Task 3's skips. Fix it there, with this test as its proof.
- [ ] **Step 6: Commit.**

```bash
git add include/TypeScript lib/TypeScript test/tester/own/own_shared_containers.ts test/tester/own/own_shared_cycle.ts test/tester/CMakeLists.txt
git commit -m "-mm=own Shared<T>: handles in containers, any and closures; cycles (spec 23.1, 23.4)"
```

---

### Task 5: measure, corpus, results, Linux

**Files:**
- Modify: the spec, a new §23.7 "Results"
- Modify: this plan (tick the boxes)

- [ ] **Step 1: Measure.** Write `$SCRATCH/shared_loop.ts`:

```ts
class N {
    v = 0;
    kids: Shared<N>[] = [];
}

function main() {
    let total = 0;
    for (let i = 0; i < 200000; i++) {
        const leaf = new Shared(new N());
        const a = new Shared(new N());
        const b = new Shared(new N());
        a.value.kids.push(leaf);
        b.value.kids.push(leaf);
        leaf.value.v = i;
        total += a.value.kids[0].value.v == b.value.kids[0].value.v ? 1 : 0;
    }

    print(total);
}
```

  - Run `pwsh test/tester/tools/measure.ps1 -Source $SCRATCH/shared_loop.ts`. It needs one plain
    test-runner run first, so that `compile.bat` exists.
  - Run it again at 20000 iterations.
  - Expected: own and rc peaks are the same at both counts, within 0.5 MB; none grows; gc stays
    bounded.
  - Record both rows.
- [ ] **Step 2: Corpus.** Run the corpus scripts (Appendix A) plain and with `OPTS=--opt`, with
  this branch's binary.
  - Expected: 453 / 593 and 451 / 593, each with the same file set as main
    (`join -t $'\t' base.tsv new.tsv | awk -F'\t' '$2 != $3'` prints nothing).
  - Make `base.tsv`/`baseopt.tsv` from main's binary first: build main once, or reuse a saved
    `tslang-main.exe`, before this branch's final build.
- [ ] **Step 3: Unchanged where it compiles.** For every file ok in `base.tsv`, compare
  `--emit=llvm -mm=own --no-default-lib <file> -o <out>.ll` from main's binary and this branch's.
  - Normalise the run-to-run names first: `([A-Za-z])_[0-9]{4,}` → `\1_N`, `\.[0-9]{6,}\.` →
    `.N.`, and the `[N x i8]` size on lines containing `FH`.
  - Expected: identical, apart from any file shown to differ between two runs of main itself.
  - Also run `--emit=llvm -mm=rc` on 50 of those files. Expected: identical. rc's routines changed
    only for `SharedType`.
- [ ] **Step 4: Spec results.** Append §23.7 "Results" in the style of §22.7:
  - what was built, with the op list, including Task 1's `SharedValueRef` amendment;
  - the tests and models;
  - the negatives with their messages;
  - the teeth (Task 3 Step 8), and what (d) showed;
  - the measure rows;
  - the corpus and IR-unchanged results;
  - the limits found: the `any` check from Task 4 Step 4, if it applies; cycles; anything a task
    recorded.

  Tick this plan's boxes, and annotate this step "committed; push and PR left to the user".
- [ ] **Step 5: Linux.** Build the branch with GCC in WSL and run
  `ctest -R "own_shared|own-err|own-no-counting|own-verify|verify-ownership-own_shared"` there.
  - The WSL clone is `~/ts/TypeScriptCompiler`, whose origin is `/mnt/i/TypeScriptCompiler`.
    Fetch this branch, and do not fetch `main`.
  - The build dir is `__build/tslang/ninja/release`.
  - Write a `.sh` into the scratchpad and run it with `wsl bash -l <path>` from PowerShell.
- [ ] **Step 6: Commit.** Do not push and do not open a PR. That is left to the user.

```bash
git add docs/superpowers/
git commit -m "-mm=own Shared<T>: measure, corpus and results (spec 23.7)"
```

---

## Appendix A: the corpus tooling (scratchpad only, never committed)

`one.sh`, the first error of one corpus file (`T` is the compiler to run):

```sh
#!/bin/sh
f="$1"; T=${T:-/i/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/tslang.exe}
out=$(cd /i/TypeScriptCompiler/tslang/test/tester/tests && timeout 120 "$T" --emit=llvm -mm=own --no-default-lib $OPTS "$(basename "$f")" -o NUL 2>&1)
rc=$?
first=$(printf '%s\n' "$out" | grep -m1 -E "error" | sed -E 's/^.*error: //' | tr -d '\r')
if [ $rc -eq 0 ]; then echo "$(basename "$f")	ok"; else echo "$(basename "$f")	${first:-rc=$rc}"; fi
```

`run.sh`, every file into `$1`:

```sh
#!/bin/sh
cd "$(dirname "$0")"
ls /i/TypeScriptCompiler/tslang/test/tester/tests/*.ts | xargs -P 12 -n 1 sh ./one.sh | sort > "${1:-results.tsv}"
echo "$(grep -c '	ok$' "${1:-results.tsv}") / $(wc -l < "${1:-results.tsv}") ok"
```
