// A function that takes the elements out of its rest parameter and keeps them keeps the parameter
// (spec 72): its callers move what they give into the array, it moves each out (`shift`, `pop`), and
// it releases the emptied array. An Array<T>-like class of objects pushes, unshifts and splices that
// way. Each value is read after churn() has reused anything freed too early.
class Item {
    constructor(public name: string) {}
}

class List<T> {
    data: T[] = [];

    push(...args: T[]) {
        while (args.length > 0) this.data.push(args.shift());
        return this.data.length;
    }

    unshift(...args: T[]) {
        while (args.length > 0) this.data.unshift(args.pop());
        return this.data.length;
    }

    // the removed elements are moved out, not read: each is swapped to the end, the first last, and
    // popped off in order
    splice(start: int, count: int, ...items: T[]) {
        const last = this.data.length - 1;
        for (let k = 0; k < count; k++) {
            for (let j = start; j < last - k; j++) {
                [this.data[j], this.data[j + 1]] = [this.data[j + 1], this.data[j]];
            }
        }

        const removed: T[] = [];
        for (let k = 0; k < count; k++) removed.push(this.data.pop());

        for (let i = 0; items.length > 0; i++) this.data.splice(start + i, 0, items.shift());

        return removed;
    }
}

function churn() {
    const hold: Item[] = [];
    for (let i = 0; i < 64; i++) hold.push(new Item("churn" + i));
}

function names(list: List<Item>) {
    let s = "";
    for (const item of list.data) s += item.name;
    return s;
}

function main() {
    const list = new List<Item>();
    list.push(new Item("a"), new Item("b"), new Item("c"));

    // a local moves into the call
    const d = new Item("d");
    list.push(d);
    churn();
    assert(names(list) == "abcd", "pushed");

    list.unshift(new Item("p"), new Item("q"));
    churn();
    assert(names(list) == "pqabcd", "unshifted");

    const removed = list.splice(2, 2, new Item("x"));
    churn();
    assert(removed.length == 2 && removed[0].name == "a" && removed[1].name == "b", "removed");
    assert(names(list) == "pqxcd", "spliced");

    print("done.");
}
