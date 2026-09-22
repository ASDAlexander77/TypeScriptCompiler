// An integer array literal that is not built for one array type keeps its `s32` elements: a const
// variable, or a literal assigned to a union like `number[] | string`. Casting it to `number[]` copied
// the elements as they were, so `number` elements were read out of `s32` data - garbage values and
// a garbage length, with only a warning.
function lengthOf(u: number[] | string) {
    return typeof u === "array" ? u.length : -1;
}

// A module-level `const` int-array literal: covers the same elementwise widening as the local
// `const c` case below, but through the global-variable codegen path (createGlobalVariable /
// adjustGlobalVariableType), not the local one (createLocalVariable / adjustLocalVariableType).
// Giving `const` array literals real identity storage (so mutating methods like .sort() share one
// heap array - see docs/const-let-storage-design.md) initially broke this: reading `moduleConst`
// back yields `AddressOf(@moduleConst) -> Load`, a shape castConstArrayToArray's local-only
// trace-back (Load -> VariableOp -> initializer) did not recognize, so it fell through to the
// generic array-to-array cast and was rejected ("element type number is not base of type s32").
// Fixed by also unwrapping AddressOf(global) -> the GlobalOp's own initializer terminator.
const moduleConst = [7, 8, 9];

function takesNumberArray(a: number[]) {
    return a[0] + a[1] + a[2];
}

function returnsNumberArray(): number[] {
    return moduleConst;
}

function main() {
    const c = [1, 2, 3];
    let fromConst: number[] = c;
    assert(fromConst.length == 3, "const variable: length");
    assert(fromConst[2] == 3, "const variable: element");

    let u: number[] | string = [1, 2];
    if (typeof u === "array") {
        assert(u.length == 2, "union initializer: length");
        assert(u[1] == 2, "union initializer: element");
    } else {
        assert(false, "union initializer: wrong branch");
    }

    let v: number[] | string;
    v = [4, 5, 6];
    if (typeof v === "array") {
        assert(v[0] + v[1] + v[2] == 15, "union assignment");
    } else {
        assert(false, "union assignment: wrong branch");
    }

    assert(lengthOf([7, 8]) == 2, "union parameter");

    // the elements are real `number`s now: fractional arithmetic works on them
    const halves: number[] = c;
    assert(halves[0] / 2 == 0.5, "number arithmetic on converted element");

    // module-level const int-array literal: assignment, parameter, and return-value shapes
    let fromModuleConst: number[] = moduleConst;
    assert(fromModuleConst.length == 3, "module const: length");
    assert(fromModuleConst[0] / 2 == 3.5, "module const: assignment, number arithmetic");

    assert(takesNumberArray(moduleConst) == 24, "module const: parameter");

    let returned = returnsNumberArray();
    assert(returned[2] / 2 == 4.5, "module const: return value, number arithmetic");

    print("done.");
}
