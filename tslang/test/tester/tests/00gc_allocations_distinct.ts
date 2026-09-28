// Two allocations of the same size are two blocks. The collector's allocators were marked so that
// the optimizer could merge two identical calls: the storage of two empty arrays became one block,
// both arrays grew through GC_realloc of it, and the heap was corrupted - here, `s += a` came out
// empty. Only with optimization, which the compile and jit tiers use.
class Point {
    x = 0;
}

function main() {
    const src: number[] = [1, 2];
    let nums: number[] = [];
    const t: number[] = [...src];
    for (const v of t) nums.push(v);
    assert(nums.length == 2, "nums");

    const a = nums[0] + ",";
    let s = "";
    s += a;
    assert(s == "1,", "append");

    let e1: number[] = [];
    let e2: number[] = [];
    e1.push(1);
    e2.push(2);
    e2.push(3);
    assert(e1.length == 1 && e1[0] == 1, "first empty array");
    assert(e2.length == 2 && e2[0] == 2, "second empty array");

    const p = new Point();
    const q = new Point();
    p.x = 1;
    q.x = 2;
    assert(p.x == 1 && q.x == 2, "two instances");

    print("done.");
}
