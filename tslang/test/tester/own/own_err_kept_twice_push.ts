// -mm=own rejects: `f` pushes both `p` and `q` into `arr`, so it keeps both, and `f(arr, n, n)` gives
// it the same object twice: the array holds two owners of one block. The second `n` is a use after
// the move into the first.
class V {
    constructor(public x: number) {}
}

class N {
    constructor(public v: V) {}
}

@noinline
function mk(i: number) {
    return new N(new V(i));
}

function churn() {
    const hold: V[] = [];
    for (let i = 0; i < 64; i++) {
        hold.push(new V(777));
    }
}

@noinline
function f(arr: N[], p: N, q: N) {
    arr.push(p);
    arr.push(q);
    return 0;
}

function main() {
    let s: number = 0;
    for (let i = 0; i < 1000; i++) {
        const arr: N[] = [];
        const n = mk(i);
        s += f(arr, n, n) + i + 2;
        churn();
        s += arr[0].v.x - i;
    }

    print(s);
    assert(s == 501500);
}
