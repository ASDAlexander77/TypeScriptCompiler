// TSLANG 190: a for-of element kept in a `let` declared outside the loop, in every memory model. Under
// -mm=own it was refused ("'b' borrows a field and cannot be stored, returned or captured"); `push` worked.
// The element moves out of the loop's `{ value, done }` record into the `let`, which gives back what it held.
class Box {
    constructor(public s: string) {}
}

function* gen(k: number) {
    yield new Box("a" + k);
    yield new Box("bb" + k);
    yield new Box("ccc" + k);
}

function digits(k: number) {
    return ("" + k).length;
}

// kept on one step only
function keepIf(k: number): string {
    let kept: Box | null = null;
    for (const b of gen(k)) {
        if (b.s.length == 2 + digits(k)) kept = b;
    }

    return kept ? kept.s : "none";
}

// the loop left once it has kept one
function keepBreak(k: number): string {
    let kept: Box | null = null;
    for (const b of gen(k)) {
        kept = b;
        if (kept.s.length == 2 + digits(k)) break;
    }

    return kept ? kept.s : "none";
}

// every one kept in turn: the one before is given back
function keepLast(k: number): string {
    let kept: Box = new Box("init" + k);
    for (const b of gen(k)) {
        kept = b;
    }

    return kept.s;
}

function main() {
    assert(keepIf(1) == "bb1", "kept on one step");
    assert(keepBreak(1) == "bb1", "kept, then break");
    assert(keepLast(1) == "ccc1", "the last one kept");

    let n = 0;
    for (let i = 0; i < 20000; i++) {
        n += keepIf(i).length + keepBreak(i).length + keepLast(i).length;
    }

    assert(n == 406670, "many times");

    print("done.");
}
