// A function that only reads its parameter borrows it.
class P {
    x: number;
    y: number;
    constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }
}

function len2(p: P) {
    return p.x * p.x + p.y * p.y;
}

function total(n: number) {
    let t: number = 0;
    for (let i = 0; i < n; i++) {
        const p = new P(i, 1);
        t += len2(p);
    }
    return t;
}

function main() {
    assert(total(10) == 295);
    print("done.");
}
