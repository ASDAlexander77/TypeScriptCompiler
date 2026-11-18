// A new array only read where it is made - its length, an element - is born uncounted with a count of
// each element, and no receiver takes it. Under rc it is held and given back at the end of its block
// (read by temp-array-literal.cmake: each array in `reads` is retained once and released once); before,
// every evaluation leaked the array and its elements. It runs in every model.

class P {
    constructor(public x: number) {}
}

function reads(a: string, b: string, i: number) {
    const n = [a, b, "x"].length;
    const e = [a, "q" + i][1];
    const m = [...[a], b].length;
    const p = [new P(i), new P(i + 1)][1].x;
    return n + e.length + m + p;
}

// an element handed out of a branch: the branch takes a count of it for the result, then gives the
// array back (read by temp-array-literal.cmake as well)
function pick(c: boolean, i: number) {
    return c ? ["c" + i][0] : "d";
}

function pickOrNull(c: boolean, i: number): string | null {
    return c ? ["n" + i, "m"][0] : null;
}

function main() {
    let total = 0;
    for (let i = 0; i < 1000; i++) {
        total += reads("s" + i, "t" + i, i);
        assert(pick(true, i) == "c" + i, "picked");
        assert(pick(false, i) == "d", "not picked");
        assert(pickOrNull(true, i) == "n" + i, "picked or null");
        assert(pickOrNull(false, i) == null, "null");
    }

    assert(total > 0, "total");
    print("done.");
}
