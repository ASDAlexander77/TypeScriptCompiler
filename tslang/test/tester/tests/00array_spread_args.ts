// push/unshift/splice with spread arguments, whose count is known only at run time, and push with
// more than one item (the op's verifier accepted exactly one).
function same(a: number[], expected: number[]) {
    if (a.length != expected.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] != expected[i]) return false;
    }

    return true;
}

function spreadOf(values: number[]) {
    let a: number[] = [1, 2];
    a.push(...values);
    return a;
}

function main() {
    let m: number[] = [];
    assert(m.push(1, 2) == 2, "push two: result");
    assert(same(m, [1, 2]), "push two");

    const extra = [3, 4, 5];

    let a: number[] = [1, 2];
    assert(a.push(...extra) == 5, "push spread: result");
    assert(same(a, [1, 2, 3, 4, 5]), "push spread");

    a.push(0, ...extra, 9);
    assert(same(a, [1, 2, 3, 4, 5, 0, 3, 4, 5, 9]), "push mixed");

    assert(same(spreadOf([]), [1, 2]), "push empty spread");
    assert(same(spreadOf([7, 8]), [1, 2, 7, 8]), "push parameter spread");

    let b: number[] = [9];
    b.unshift(...extra);
    assert(same(b, [3, 4, 5, 9]), "unshift spread");
    b.unshift(7, ...[8]);
    assert(same(b, [7, 8, 3, 4, 5, 9]), "unshift mixed");

    let c: number[] = [1, 2, 3];
    c.splice(1, 1, ...extra);
    assert(same(c, [1, 3, 4, 5, 3]), "splice spread");
    c.splice(-1, 1, ...[10, 11]);
    assert(same(c, [1, 3, 4, 5, 10, 11]), "splice negative start");
    c.splice(100, 0, ...[12]);
    assert(same(c, [1, 3, 4, 5, 10, 11, 12]), "splice start past the end");
    c.splice(-100, 2, ...[0]);
    assert(same(c, [0, 4, 5, 10, 11, 12]), "splice start before the beginning");

    const ints = [1, 2];
    let nums: number[] = [];
    nums.push(...ints);
    assert(same(nums, [1, 2]), "push s32 spread into number[]");

    let ss: string[] = [];
    ss.push(...["a"], "b");
    assert(ss.length == 2, "strings: length");
    assert(ss[0] == "a", "strings: first");
    assert(ss[1] == "b", "strings: second");

    print("done.");
}
