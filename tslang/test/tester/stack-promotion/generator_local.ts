// A generator's body is re-entered per next(): no promotion in version 1.
class P {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

function* gen(n: number) {
    for (let i = 0; i < n; i++) {
        const p = new P(i);
        yield p.x;
    }
}

function main() {
    let t: number = 0;
    for (const v of gen(5)) {
        t += v;
    }
    assert(t == 10);
    print("done.");
}
