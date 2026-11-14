// TSLANG 83: a row of a module-level array of arrays bound to a local (`const meta = POOL[idx]`): the
// rows are constant arrays the literal makes, not made immortal, so the global is no keeper and the
// local borrows the row. Taken for a keeper's immortal element, each call freed it: a double free from
// the second call (a hang at exit, or a crash).
const POOL = [[1000.0, 0.001], [500.0, 0.002]];
const NAMES = ["a", "bb"];

function build(idx: int): number {
    const meta = POOL[idx];
    return meta[0];
}

function name(idx: int): string {
    const n = NAMES[idx];
    return n;
}

function main() {
    let total = 0.0;
    let length = 0;
    for (let i = 0; i < 1000; i++) {
        total += build(i % 2);
        length += name(i % 2).length;
    }

    assert(total == 750000 && POOL[0][0] == 1000 && POOL[1][0] == 500, "pool");
    assert(length == 1500, "names");
    print("done.");
}
