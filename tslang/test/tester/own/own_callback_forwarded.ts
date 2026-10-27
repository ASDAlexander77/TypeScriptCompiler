// A callback given on through a parameter (spec 43): the functions passed for the parameter it is
// given from are passed for the one it is given to, so a call through it - with two elements of an
// array the callback only reads - knows it releases nothing. An optional callback is tested for
// being given, and one not given passes nothing. Given on once, twice, and to the function itself;
// an optional one called itself.
// Each case churns between steps with blocks the size of the ones read, so a block freed by mistake
// is reused.
class Item {
    v: number[] = [];
    constructor(public key: int) {}
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new Item(999);
        c.v.push(99);
        junk.push(c);
    }
}

function made(n: int): Item[] {
    const r: Item[] = [];
    for (let i = n - 1; i >= 0; i--) {
        const c = new Item(i);
        c.v.push(i);
        r.push(c);
    }

    return r;
}

function selectionSort<T>(a: T[], compare: (x: T, y: T) => int) {
    const len = a.length;
    for (let i = 0; i < len - 1; ++i) {
        for (let j = i + 1; j < len; ++j) {
            if (compare(a[i], a[j]) > 0) {
                [a[i], a[j]] = [a[j], a[i]];
            }
        }
    }
}

// as DefaultLib's sort: an optional comparison, given on when given
function sortIt<T>(this: T[], callbackfn?: (x: T, y: T) => int) {
    if (callbackfn) {
        selectionSort(this, callbackfn);
    } else {
        selectionSort(this, (a: T, b: T) => 0);
    }
}

function each<T>(a: T[], f: (v: T) => void) {
    for (let i = 0; i < a.length; i++) {
        f(a[i]);
    }
}

function eachTwice<T>(a: T[], f: (v: T) => void) {
    each(a, f);
    each(a, f);
}

function eachDown<T>(a: T[], i: int, f: (v: T) => void) {
    if (i < 0) {
        return;
    }

    f(a[i]);
    eachDown(a, i - 1, f);
}

// an optional callback called itself, once tested
function countDescents<T>(a: T[], compare?: (x: T, y: T) => int) {
    let n = 0;
    for (let i = 0; i < a.length - 1; ++i) {
        if (compare) {
            if (compare(a[i], a[i + 1]) > 0) n++;
        }
    }

    return n;
}

let total: number = 0;

function add(c: Item) {
    total += c.key + c.v[0];
}

function main() {
    let ok = 0;
    for (let round = 0; round < 1000; round++) {
        const a = made(5);
        a.sortIt((p: Item, q: Item) => p.key - q.key);
        churn();
        if (a[0].key == 0 && a[2].key == 2 && a[4].key == 4 && a[4].v[0] == 4) ok++;

        const b = made(3);
        b.sortIt();
        churn();
        if (b[0].key == 2 && b[2].key == 0) ok++;

        total = 0;
        const c = made(4);
        eachTwice(c, add);
        churn();
        if (total == 24 && c[3].v[0] == 0) ok++;

        total = 0;
        eachDown(c, 3, add);
        churn();
        if (total == 12 && c[0].key == 3) ok++;

        const d = made(4);
        const descents = countDescents(d, (p: Item, q: Item) => p.key - q.key) + countDescents(d);
        churn();
        if (descents == 3 && d[3].v[0] == 0) ok++;
    }

    assert(ok == 1000 * 5, "ok");
    print("done.");
}
