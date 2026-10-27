// -mm=own rejects: `sortIt` gives its comparison on to `selectionSort` (spec 43), so the function
// passed to `sortIt` is the one `selectionSort` calls - and it replaces a global array, releasing
// what it held. `selectionSort` calls it with two elements of an array that may be that global, and
// reads them again after the call.
class Item {
    constructor(public key: int) {}
}

let pool: Item[] = [];

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

function sortIt<T>(this: T[], callbackfn?: (x: T, y: T) => int) {
    if (callbackfn) {
        selectionSort(this, callbackfn);
    } else {
        selectionSort(this, (a: T, b: T) => 0);
    }
}

function main() {
    const items = [new Item(3), new Item(1), new Item(2)];
    pool = [new Item(7)];
    items.sortIt((p: Item, q: Item) => {
        pool = [];
        return p.key - q.key;
    });
    print(items[0].key, pool.length);
}
