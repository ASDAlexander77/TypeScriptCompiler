// A nested integer array literal that is not built for one array type - a const variable, or a
// literal meeting a union - keeps `s32` elements in its inner arrays. Casting it to `number[][]`
// copied those as they were: inner elements came out as garbage, and the union case did not compile
// at all ("'ts.Cast' op ... can't be stored in").
function main() {
    const c = [[1, 2], [3, 4]];
    let fromConst: number[][] = c;
    assert(fromConst.length == 2, "const variable: outer length");
    assert(fromConst[0].length == 2, "const variable: inner length");
    assert(fromConst[1][1] == 4, "const variable: inner element");
    assert(fromConst[0][0] / 2 == 0.5, "const variable: element is a number");

    let u: number[][] | string = [[5, 6]];
    if (typeof u === "array") {
        assert(u.length == 1, "union: outer length");
        assert(u[0][1] == 6, "union: inner element");
    } else {
        assert(false, "union: wrong branch");
    }

    // three levels deep
    const deep = [[[7, 8]]];
    let fromDeep: number[][][] = deep;
    assert(fromDeep[0][0][1] == 8, "three levels");

    print("done.");
}
