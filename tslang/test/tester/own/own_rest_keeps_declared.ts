// A rest parameter whose function says it keeps it (`@keeps("args")`) is released when emptied, as
// one inferred is (spec 72): its callers move what they give into the array, it moves each element
// out, and it releases the array at its exits. A declared keep skipped that release, and every call
// leaked its array; no run can see that leak, so the IR is read too. Each value is read after churn()
// has reused anything freed too early.
class Item {
    constructor(public name: string) {}
}

class List {
    data: Item[] = [];

    @keeps("args")
    push(...args: Item[]) {
        while (args.length > 0) this.data.push(args.shift());
        return this.data.length;
    }
}

function churn() {
    const hold: Item[] = [];
    for (let i = 0; i < 64; i++) hold.push(new Item("churn" + i));
}

function main() {
    const list = new List();
    list.push(new Item("a"), new Item("b"));

    // a local moves into the call
    const c = new Item("c");
    list.push(c);
    churn();

    let s = "";
    for (const item of list.data) s += item.name;
    assert(s == "abc", "pushed");

    print("done.");
}
