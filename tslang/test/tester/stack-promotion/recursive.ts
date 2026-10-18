// A recursive function gets no promotion: every frame would carry the object.
class Pair {
    a: number;
    b: number;
    constructor(a: number, b: number) {
        this.a = a;
        this.b = b;
    }
}

function depthSum(n: number): number {
    if (n == 0) {
        return 0;
    }

    const p = new Pair(n, 1);
    return p.a + p.b - 1 + depthSum(n - 1);
}

function main() {
    assert(depthSum(1000) == 500500);
    print("done.");
}
