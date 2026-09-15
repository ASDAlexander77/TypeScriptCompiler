// An integer array literal that is not built for one array type keeps its `s32` elements: a const
// variable, or a literal assigned to a union like `number[] | string`. Casting it to `number[]` copied
// the elements as they were, so `number` elements were read out of `s32` data - garbage values and
// a garbage length, with only a warning.
function lengthOf(u: number[] | string) {
    return typeof u === "array" ? u.length : -1;
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

    print("done.");
}
