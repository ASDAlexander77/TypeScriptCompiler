// -mm=own: one object pushed twice in one push would have two owners in the array
class N {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

@noinline
function mk(i: number): N {
    return new N(i);
}

function main() {
    const n: N = mk(1);
    const a: N[] = [];
    a.push(n, n);
    print("x", a[0].x);
}
