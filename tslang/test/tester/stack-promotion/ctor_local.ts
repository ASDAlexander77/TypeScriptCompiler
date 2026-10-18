// The constructor only assigns fields: it borrows `this`, so the object stays on the stack.
class P {
    x: number;
    y: number;
    constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }
}

function sum(n: number) {
    let t: number = 0;
    for (let i = 0; i < n; i++) {
        const p = new P(i, i + 1);
        t += p.x + p.y;
    }
    return t;
}

function main() {
    assert(sum(1000) == 1000000);
    print("done.");
}
