// -mm=own: one object given twice to an array literal would have two owners, and the array would
// free it twice when it goes (fuzz 953312: the heap then hung at exit on Linux)
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
    const a: N[] = [mk(2), n, n];
    print("x", a[1].x);
}
