// regression test: mutating a `const` array through an alias (element writes, and a
// helper function that reorders elements in place) silently did nothing - or crashed -
// both at module scope and inside a function body.
//
// Root cause: a `const` array literal resolves to `!ts.const_array<T,N>` (a
// compile-time-constant snapshot), not the mutable heap-backed `!ts.array<T>` a
// `let` array gets. `registerVariable`/`adjustLocalVariableType` never gave a plain
// `const` binding real storage or widened its type the way `let` does (only a
// narrower case -- a bound-method-carrying tuple, e.g. a generator wrapper -- forced
// real storage). So every mutating use had to `ts.Cast` a fresh, disposable
// `!ts.array<T>` copy out of the pristine const_array, while direct element
// reads/writes indexed the immutable original directly: mutation through a passed-in
// array parameter (or a chain of them) was lost, and a direct element write into the
// immutable original segfaulted (0xC0000005). `let` arrays, and non-array mutating
// const bindings, were unaffected.
//
// This deliberately avoids Array.prototype.sort()/.reverse() (the methods in the
// original bug report): those are default-lib extension methods, and this suite's
// `test-runner` always passes `--no-default-lib`, so it writes its own tiny in-place
// sort out of plain element reads/writes/function calls instead - the same shape of
// mutation-through-identity that Array.sort()/.reverse() exercise, without the
// default-lib dependency. See docs/const-let-storage-design.md and the const-array
// notes next to processConstRef/adjustLocalVariableType/adjustGlobalVariableType in
// MLIRGenImpl.h.
const moduleArr: number[] = [3, 1, 2];

function swap(a: number[], i: number, j: number) {
    const t = a[i];
    a[i] = a[j];
    a[j] = t;
}

// tiny in-place bubble sort over 3 elements, written out longhand instead of a loop
// so this test has no dependency beyond plain control flow and array indexing.
function sort3(a: number[]) {
    if (a[0] > a[1]) { swap(a, 0, 1); }
    if (a[1] > a[2]) { swap(a, 1, 2); }
    if (a[0] > a[1]) { swap(a, 0, 1); }
}

function localMutation() {
    const arr: number[] = [3, 1, 2];
    sort3(arr);
    assert(arr[0] == 1 && arr[1] == 2 && arr[2] == 3, "local const: sort via aliased element writes");

    arr[0] = 9;
    assert(arr[0] == 9 && arr[1] == 2 && arr[2] == 3, "local const: direct element write");
}

function main() {
    localMutation();

    sort3(moduleArr);
    assert(moduleArr[0] == 1 && moduleArr[1] == 2 && moduleArr[2] == 3, "module const: sort via aliased element writes");

    moduleArr[0] = 9;
    assert(moduleArr[0] == 9 && moduleArr[1] == 2 && moduleArr[2] == 3, "module const: direct element write");

    print("done.");
}
