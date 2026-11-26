// TSLANG 208 (the "not yet" of 207), every model.

// a literal held by a variable first, whose function member returns an integer, given as an interface whose member
// returns a number: "field "produce" not matching type" (`() => s32` is not `() => number`)
interface Options<Data> {
    produce: () => Data;
}

function one() {
    return 1;
}

function read(o: Options<number>) {
    return o.produce();
}

// a tuple or object literal given to a callee that keeps it: under -mm=own "this function does not own it"
class K {
    items: [string, number][] = [];
    objs: { s: string; n: number }[] = [];

    keepOne(t: [string, number]) {
        this.items.push(t);
    }

    keepObj(t: { s: string; n: number }) {
        this.objs.push(t);
    }
}

function lengthOf(t: [string, number]) {
    return t[0].length + t[1];
}

function main() {
    const t = { produce: () => 45 };
    const o: Options<number> = t;
    assert(o.produce() / 2 == 22.5);
    assert(read(t) / 2 == 22.5);

    const named = { produce: one };
    const p: Options<number> = named;
    assert(p.produce() / 2 == 0.5);

    const k = new K();
    for (let i = 0; i < 3; i++) {
        k.keepOne(["a" + i, i]);
        k.keepObj({ s: "b" + i, n: i });
        k.keepOne((["c" + i, i + 10]));
    }

    assert(k.items.length == 6 && k.items[4][0] == "a2" && k.items[5][0] == "c2" && k.items[5][1] == 12);
    assert(k.objs.length == 3 && k.objs[2].s == "b2" && k.objs[2].n == 2);

    // a callee that only reads it
    let sum: number = 0;
    for (let i = 0; i < 3; i++) {
        sum += lengthOf(["x" + i, 1]);
    }

    assert(sum == 9);

    print("done.");
}
