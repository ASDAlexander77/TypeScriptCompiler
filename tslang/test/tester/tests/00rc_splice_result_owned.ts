// -mm=rc: the array `splice` gives back arrives owned - its receiver takes the reference over and does not
// retain - but its header block was born with no reference at all. A method that kept the result in a local and
// returned it (`const removed = this.data.splice(...); return removed;`, the default library's Array<T>.splice)
// retained it to 1 for the return and released the local to 0 on the way out: the caller got a freed array, and
// the next allocations wrote over it (array_class_splice.ts crashed with heap corruption under the debug JIT on
// Linux, and this one crashes on Windows).
class Item {
    constructor(public name: string) {}
}

class Arr<T> {
    data: T[] = [];

    take(start: int, count: int): T[] {
        const removed = this.data.splice(start, count);
        for (let i = 0; i < 1; i++) {}
        return removed;
    }

    takeIf(start: int, count: int, early: boolean): T[] {
        const removed = this.data.splice(start, count);
        if (early) {
            return removed;
        }

        return removed;
    }
}

function main() {
    const a = new Arr<Item>();
    for (let i = 0; i < 6; i++) a.data.push(new Item("i" + i));

    let ok = 0;
    for (let n = 0; n < 1000; n++) {
        const removed = n % 2 == 0 ? a.take(1, 2) : a.takeIf(1, 2, n % 4 == 1);
        // the same size of block as the array's header, made right after: a freed header is handed out again
        const other: Item[][] = [];
        for (let k = 0; k < 8; k++) other.push([new Item("z" + k)]);
        if (removed.length == 2 && removed[0].name.length == 2 && removed[1].name.length == 2) ok++;
        a.data.splice(1, 0, removed[0], removed[1]);
    }

    assert(ok == 1000, "every result read back as it was taken");
    assert(a.data.length == 6);

    // a result nobody keeps is given back where it is made
    for (let n = 0; n < 100; n++) {
        a.data.splice(0, 1);
        a.data.push(new Item("p" + n));
    }

    assert(a.data.length == 6);

    print("done.");
}
