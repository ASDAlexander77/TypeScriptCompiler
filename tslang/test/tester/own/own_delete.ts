// -mm=own: `delete` ends the owner the expression names - a local or a field that owns its value is released
// and cleared, as an assignment releases what it replaces; a value no storage holds goes where its block ends.
function churn(): number {
    let n = 0;
    for (let i = 0; i < 400; i++) {
        const s = "filler " + i;
        n = n + s.length;
    }

    return n;
}

class Leaf {
    name: string = "leaf";
    constructor(public v: number) {}
}

class Holder {
    child?: Leaf;
    other: Leaf = new Leaf(2);

    drop() {
        delete this.other;
    }
}

function deleteALet() {
    let c = new Leaf(1);
    c.name = "let" + c.v;
    assert(c.name == "let1");
    delete c;
    churn();
}

function deleteAConst() {
    const c = new Leaf(3);
    assert(c.v == 3);
    delete c;
    churn();
}

function deleteATemporary() {
    delete new Leaf(4);
    churn();
}

function deleteAfterAMove() {
    let c = new Leaf(5);
    const h = new Holder();
    h.child = c;
    // what `c` held is the field's now: nothing is freed here
    delete c;
    churn();
    assert(h.child!.v == 5, "moved out before the delete");
}

function deleteAField() {
    const h = new Holder();
    h.child = new Leaf(6);
    assert(h.other.v == 2);
    h.drop();
    churn();
    assert(h.child.v == 6, "the other field is untouched");
    const g = new Holder();
    delete g.other;
    churn();
}

function deleteOnOnePath() {
    let total = 0;
    for (let i = 0; i < 100; i++) {
        let c = new Leaf(i);
        if (i % 2 == 0) {
            total += c.v;
            delete c;
        } else {
            total += c.v;
        }
    }

    churn();
    assert(total == 4950);
}

function deleteAnArray() {
    let a: Leaf[] = [new Leaf(8), new Leaf(9)];
    assert(a[1].v == 9);
    delete a;
    churn();
}

function deleteThenAssign() {
    let c = new Leaf(10);
    delete c;
    c = new Leaf(11);
    churn();
    assert(c.v == 11);
}

function main() {
    deleteALet();
    deleteAConst();
    deleteATemporary();
    deleteAfterAMove();
    deleteAField();
    deleteOnOnePath();
    deleteAnArray();
    deleteThenAssign();
    print("done.");
}
