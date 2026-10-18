// `P(...)` without `new` is the same as `new P(...)`: same constructor call, same object.
class P {
    x: number;
    y: number;
    constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }
    toString() {
        return "P";
    }
}

function make() {
    return P(5, 6);
}

function main() {
    const a = P(1, 2);
    const b = new P(3, 4);
    assert(a.x + a.y == 3);
    assert(b.x + b.y == 7);
    assert(make().y == 6);
    assert(a instanceof P);
    print("done.");
}
