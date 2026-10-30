// -mm=own: a read through an interface's field that a getter may fill, put in a `let` that is only read -
// across a loop, out of the read's block - is freed at the function's returns, on the getter's path only
// (OwnershipInferencePass::localHeldToExits). It leaked before: two million calls kept two million Boxes
// (280 MB, 4.3 MB now). Through a plain field the same read is a borrow, and is not freed. Values are read
// after churn() has reused anything freed too early.
class Box {
    data: number[] = [1, 2, 3];
    constructor(public n: number) {}
}

interface I {
    item: Box;
}

class Fresh implements I {
    private k = 0;
    get item() { this.k = this.k + 1; return new Box(10 * this.k); }
    set item(x: Box) { this.k = x.n; }
}

class Plain implements I {
    item = new Box(2);
}

function churn() {
    const junk: Box[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new Box(999));
    }
}

function heldInLet(i: I) {
    let x = i.item;
    let total: number = 0;
    for (let k = 0; k < 3; k++) {
        churn();
        total = total + x.data.length;
    }

    return total + x.n;
}

function main() {
    const f: I = new Fresh();
    const p: I = new Plain();
    let s: number = 0;
    for (let k = 0; k < 4; k++) {
        s = s + heldInLet(f) + heldInLet(p);
        churn();
    }

    assert(s == (9 + 10) + (9 + 20) + (9 + 30) + (9 + 40) + 4 * (9 + 2), "sum");
    assert(p.item.n == 2 && p.item.data.length == 3, "plain item kept");
    print("done.");
}
