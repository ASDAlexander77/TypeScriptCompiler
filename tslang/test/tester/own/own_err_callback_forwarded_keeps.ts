// -mm=own rejects: `sortIt` gives its comparison on to `selectionSort` (spec 43), so the function
// passed to `sortIt` is the one `selectionSort` calls - and it keeps its first argument for good, in
// a global array. Its callers must make that argument immortal, and `selectionSort` calls it through
// a value, which cannot.
class Item {
    constructor(public key: int) {}
}

let kept: Item[] = [];

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
    items.sortIt((p: Item, q: Item) => {
        kept.push(p);
        return p.key - q.key;
    });
    print(items[0].key, kept.length);
}
