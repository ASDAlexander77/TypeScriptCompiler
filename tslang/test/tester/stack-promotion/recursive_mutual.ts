// Functions on a cycle of calls keep their objects on the heap; a function that only calls into
// the cycle is not on it.
class P {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

function isEven(n: number): boolean {
    const p = new P(n);
    return p.x == 0 ? true : isOdd(p.x - 1);
}

function isOdd(n: number): boolean {
    const q = new P(n);
    return q.x == 0 ? false : isEven(q.x - 1);
}

function check(n: number) {
    const r = new P(n);
    return isEven(r.x);
}

function main() {
    assert(check(10));
    assert(!check(7));
    print("done.");
}
