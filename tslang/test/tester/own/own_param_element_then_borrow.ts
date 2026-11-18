// A parameter put into an array literal, then given one of the array's elements (fuzz 958275): the
// array owns the argument, so the function keeps it - its caller gives it up. Taken for a borrow, the
// array freed the caller's object and the caller read it after.
class V {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

function churn() {
    const j: V[] = [];
    for (let i = 0; i < 64; i++) j.push(new V(999));
}

function intoArray(p: V): number {
    const a: V[] = [new V(1), p];
    p = a[0];
    return p.x + a[1].x;
}

function main() {
    for (let i = 0; i < 300; i++) {
        assert(intoArray(new V(i)) == 1 + i, "into array");
        churn();
    }
    print("done.");
}
