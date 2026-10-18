// A method call, a comparison, instanceof and printing through its own toString: none of these
// let the object out.
class P {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
    twice() {
        return this.x * 2;
    }
    toString() {
        return "P";
    }
}

function check(i: number) {
    const p = new P(i);
    const q = new P(i);
    assert(p.twice() == i * 2);
    assert(p !== q);
    assert(p instanceof P);
    print(p.toString());
}

function main() {
    for (let i = 0; i < 3; i++) {
        check(i);
    }
    print("done.");
}
