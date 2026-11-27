// TSLANG 210, every model: under -mm=own a value moved in a `try` body after a statement that may throw, with more
// after it (`try { mayFail(i); const t = [...]; k.keepOne(t); mayFail(i + 1); }`), was refused: "moved here on some
// paths only". The release on the unwind leg is reached from before the move and after it.
class K {
    items: [string, number][] = [];
    objs: { s: string; n: number }[] = [];

    keepOne(t: [string, number]) {
        this.items.push(t);
    }

    keepObj(o: { s: string; n: number }) {
        this.objs.push(o);
    }
}

function mayFail(i: number) {
    if (i % 5 == 4) throw "mayFail";
}

function lengthOf(t: [string, number]) {
    return t[0].length;
}

function main() {
    const k = new K();
    let caught = 0;
    let sum: number = 0;
    for (let i = 0; i < 10; i++) {
        try {
            mayFail(i);
            const t: [string, number] = ["b" + i, i];
            k.keepOne(t);
            mayFail(i + 1);
        } catch (e) {
            caught++;
        }

        try {
            mayFail(i);
            k.keepOne(["a" + i, i]);
            k.keepObj({ s: "o" + i, n: i });
            sum += lengthOf(["x" + i, 1]);
            mayFail(i + 1);
        } catch (e) {
            caught++;
        }
    }

    // i = 4 and 9 throw before the moves, i = 3 and 8 after them
    assert(caught == 8, "caught");
    assert(k.items.length == 16 && k.objs.length == 8, "kept");
    assert(k.items[0][0] == "b0" && k.items[1][0] == "a0" && k.items[15][0] == "a8", "items");
    assert(k.objs[7].s == "o8" && sum == 16, "objs");
    print("done.");
}
