// What a generator body makes and what it yields is given back.
//
// OwnedReturnConsumptionPass used to skip whole generators: the state machine cuts their blocks
// at every resume point later, so a release at the end of a block could stop being dominated by
// the definition. Under rc and own `yield ("a" + i).length` leaked the number's string and the
// concatenation on every call: 500000 runs of `lengths` below grew by about 165 MB.
//
// A temporary is released in front of the first yield, resume point or statement holding one
// after it (`branchYield`: the `if` with a yield inside), where the definition still dominates.
// A temporary used after that point would leak, but none is: MLIRGen keeps a generator's locals
// in its state object.
//
// A yielded value: under rc a yield retains its `{ value, done }` record the way a return does,
// so the record owns what it holds, and the caller takes the record over - `next` is read out of
// a field of the state object that only ever holds it. Before, `yield "s" + i` and
// `yield new Item(i)` kept the reference they were born with and nobody gave it back (50 MB over
// 500000 runs). Under own the value moves into the record and the caller owns it.
//
// test-rc-generator-temporaries reads the IR for which is which.

// A function whose returned string the pass cannot see retained. With it in the module, a call
// through a plain value is not taken to return owned for that reason alone, so `main`'s calls of
// `next` are taken over only because the field they are read from names the generator's `next`.
declare function strerror(errnum: s32): string;

class Item {
    constructor(public n: number) {}
}

function* lengths() {
    let i = 0;
    while (true) {
        i++;
        if (i > 3) break;
        yield ("a" + i).length;
    }
}

function* strings() {
    let i = 0;
    while (true) {
        i++;
        if (i > 3) break;
        yield "s" + i;
    }
}

function* items() {
    for (let i = 1; i <= 3; i++) {
        yield new Item(i);
    }
}

function* branchYield() {
    let i = 0;
    while (i < 3) {
        i++;
        const n = ("b" + i).length;
        if (n > 0) {
            yield n;
        }
    }
}

// a temporary between two yields belongs to the stretch after the first one's resume point
function* twoSegments() {
    yield ("x" + 1).length;
    yield ("yy" + 22).length;
}

function main() {
    let total = 0;
    for (let n = 0; n < 1000; n++) {
        for (const v of lengths()) total += v;
    }

    assert(total == 6000, "lengths");

    let joined = "";
    for (const s of strings()) joined += s;
    assert(joined == "s1s2s3", "strings");

    // kept past the next call: the caller's own reference
    const it = strings();
    const first = it.next();
    const second = it.next();
    assert(first.value == "s1" && second.value == "s2", "strings by hand");

    let itemTotal = 0;
    for (const item of items()) itemTotal += item.n;
    assert(itemTotal == 6, "items");

    let branchTotal = 0;
    for (const v of branchYield()) branchTotal += v;
    assert(branchTotal == 6, "branchYield");

    let segments = 0;
    for (const v of twoSegments()) segments += v;
    assert(segments == 6, "twoSegments");

    assert(strerror(2).length > 0, "strerror");

    print("done.");
}
