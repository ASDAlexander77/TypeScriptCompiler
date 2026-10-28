// `compare && compare(a, b)` of an optional callback: a function is never falsy, so the `&&` is
// `undefined | number` (spec 53), and no `boolean | function` union holds the callback. Under own the
// call is one through the parameter, known by the family of functions given for it (spec 43): it
// releases nothing of the two elements it is given. Each step churns; every model runs it.
class Item {
    constructor(public n: number) {}
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Item(999));
}

function bubble(a: Item[], compare?: (x: Item, y: Item) => number) {
    for (let i = 0; i < a.length; i++) {
        for (let j = 0; j + 1 < a.length - i; j++) {
            const r = compare && compare(a[j], a[j + 1]);
            if (r !== undefined && r > 0) {
                [a[j], a[j + 1]] = [a[j + 1], a[j]];
            }
        }
    }
}

function byN(x: Item, y: Item) {
    return x.n - y.n;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        const a = [new Item(3), new Item(1), new Item(2)];
        bubble(a, byN);
        churn();
        if (a[0].n == 1 && a[2].n == 3) ok++;

        const b = [new Item(3), new Item(1)];
        bubble(b);
        churn();
        if (b[0].n == 3) ok++;
    }

    assert(ok == 300 * 2, "ok");
    print("done.");
}
