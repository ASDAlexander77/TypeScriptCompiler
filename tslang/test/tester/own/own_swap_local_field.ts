// `[data, this.data] = [this.data, data]`, an exchange of a local and a field or an element (spec
// 77): the field is given the local's value and the local the field's, so a value is taken out of a
// place the way it is under every other model - for a string, a class instance, a `T | undefined` of
// either (and `<T>data` of what came out), an element. Each case churns between steps with blocks
// the size of the ones taken, so a block freed by mistake is reused and read as garbage.
class Item {
    v: number[] = [];
    constructor(public n: number) {}
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new Item(999);
        c.v.push(99);
        junk.push(c);
    }
}

// a slot a value is put into and later taken back out of
class Slot<T> {
    constructor(public data: T | undefined) {}

    take(): T {
        let data: T | undefined = undefined;
        [data, this.data] = [this.data, data];
        return <T>data;
    }

    put(value: T) {
        let data: T | undefined = value;
        [data, this.data] = [this.data, data];
    }
}

// a field that always holds one: the local gives the field its own
class Holder {
    constructor(public item: Item) {}

    exchange(): Item {
        let item = new Item(0);
        [item, this.item] = [this.item, item];
        return item;
    }
}

function takeElement(items: Item[], index: int): Item {
    let item = new Item(-1);
    [item, items[index]] = [items[index], item];
    return item;
}

function main() {
    const text = new Slot<string>("a");
    churn();
    assert(text.take() == "a", "string taken");
    assert(text.data === undefined, "string slot empty");

    const slot = new Slot<Item>(new Item(1));
    churn();
    const taken = slot.take();
    churn();
    assert(taken.n == 1, "item taken");
    assert(slot.data === undefined, "item slot empty");
    slot.put(new Item(2));
    churn();
    assert(slot.take().n == 2, "item put back and taken");

    const holder = new Holder(new Item(3));
    churn();
    const old = holder.exchange();
    churn();
    assert(old.n == 3, "old item out");
    assert(holder.item.n == 0, "new item in");

    const items = [new Item(4), new Item(5)];
    churn();
    const out = takeElement(items, 1);
    churn();
    assert(out.n == 5, "element taken");
    assert(items[1].n == -1, "element replaced");
    assert(items[0].n == 4, "other element kept");

    print("done.");
}
