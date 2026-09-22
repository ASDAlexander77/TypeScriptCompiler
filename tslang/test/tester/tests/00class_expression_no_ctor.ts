let ctorCalls = 0;

let Point = class {
    constructor(public x: number, public y: number) { ctorCalls++; }
};

function main() {
    assert(ctorCalls == 0, "class expression must not call the constructor");
    const p = new Point(3, 4);
    assert(ctorCalls == 1);
    assert(p.x == 3 && p.y == 4);
    print("done.");
}
