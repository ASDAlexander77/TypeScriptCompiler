// -mm=own: an empty array literal given straight to a call (`new Order(i, [])`) is the new empty array it is: moved into
// a callee that keeps it - the constructor storing it in a field - and given back after one that does not (`ln([])`).
// Owned as the constant-array copy it was made from, it was given back after the keeping constructor too, and the
// object freed it again (the DefaultLib test worker_objects, `worker.postMessage(new Order(-1, []))`, failed on wasm
// with a free of a freed block). The same program under gc and rc.
class Order {
    constructor(public id: int, public items: string[]) {}
}

function ln(xs: string[]): number {
    return xs.length + 1;
}

function count(o: Order): number {
    return o.items.length;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 1000; i++) {
        const o = new Order(i, []);
        o.items.push("a" + i);
        t += count(o) + ln([]);
        const kept = [new Order(i, []), new Order(i + 1, [])];
        t += kept.length + kept[1].items.length;
    }

    print(t);
    assert(t == 4000, "total");
    print("done.");
}
