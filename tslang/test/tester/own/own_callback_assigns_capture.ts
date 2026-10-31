// -mm=own: a callback that gives a captured object a new value (`last = new Box(b.n)`), run by a method
// that walks the array a field holds. The assignment was a drop of anything, so the method's borrow of
// each element was "used after it may be released". It frees what only the captured variable owns
// (OWN_CAPTURED_CELL_POSITION): a borrow read through a captured variable may be under it, nothing else.
// Values are read after churn() has reused anything freed too early. The same program under every model.
class Box {
    constructor(public n: number) {}
}

class Bag {
    items: Box[] = [new Box(1), new Box(2), new Box(3)];

    forEach(f: (b: Box) => void) {
        for (const b of this.items) {
            f(b);
            churn();
        }
    }
}

function churn() {
    const junk: Box[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new Box(999));
    }
}

function main() {
    const bag = new Bag();

    let total: number = 0;
    bag.forEach(b => { total = total + b.n; });

    // the last one, replaced on each call
    let last = new Box(0);
    bag.forEach(b => { last = new Box(b.n * 10); });
    churn();

    // the largest, replaced on some calls only
    let top = new Box(0);
    bag.forEach(b => { if (b.n > top.n) top = new Box(b.n); });
    churn();

    assert(total == 6, "total");
    assert(last.n == 30, "last");
    assert(top.n == 3, "top");
    assert(bag.items.length == 3 && bag.items[0].n == 1 && bag.items[2].n == 3, "items kept");
    print("done.");
}
