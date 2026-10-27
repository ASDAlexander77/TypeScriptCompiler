// An array of handles, walked by functions given a callback, through a class that holds the
// array in a field. A `for...of` local over handles is a count of its own, and its release is
// never the last (spec 51): the array still holds one, so the walk drops nothing the caller
// borrows. A handle given to a callback gets a count of its own for the call, and one yielded on
// some paths only (`if (f(v)) yield v`) goes into the record `next()` returns with one (spec 50).
// A handle spliced in is counted, as one pushed is. Each step churns, so a payload freed too early
// reads garbage; every model runs it.
class Item {
    constructor(public name: string) {}

    toString() {
        return this.name;
    }
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Item("junk"));
}

function every(a: Shared<Item>[], f: (v: Shared<Item>) => boolean) {
    for (const v of a) if (!f(v)) return false;
    return true;
}

function* matching(a: Shared<Item>[], f: (v: Shared<Item>) => boolean) {
    for (const v of a) if (f(v)) yield v;
}

class Items {
    constructor(private data: Shared<Item>[] = []) {}

    push(v: Shared<Item>) {
        this.data.push(v);
    }

    insert(at: int, v: Shared<Item>) {
        this.data.splice(at, 0, v);
    }

    all(f: (v: Shared<Item>) => boolean) {
        return every(this.data, f);
    }

    join() {
        let s = "";
        for (const v of this.data) s += v.value.name;
        return s;
    }

    @borrows("this", "f")
    matching(f: (v: Shared<Item>) => boolean) {
        return matching(this.data, f);
    }
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        const items = new Items();
        items.push(new Shared<Item>(new Item("a")));
        items.push(new Shared<Item>(new Item("bb")));
        items.insert(1, new Shared<Item>(new Item("c")));
        churn();
        if (items.join() == "acbb") ok++;
        if (items.all((v) => v.value.name.length > 0)) ok++;
        churn();

        const kept: Shared<Item>[] = [];
        for (const v of items.matching((v) => v.value.name.length == 1)) kept.push(v);
        churn();
        if (kept.length == 2 && kept[1].value.name == "c") ok++;
    }

    assert(ok == 300 * 3, "ok");
    print("done.");
}
