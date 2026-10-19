// A temporary made inside a generator body is given back before the yield that ends the call.
//
// OwnedReturnConsumptionPass used to skip whole generators: the state machine cuts their blocks
// at every resume point later, so a release at the end of a block could stop being dominated by
// the definition. Under rc and own `yield ("a" + i).length` leaked the number's string and the
// concatenation on every call: 500000 runs of `lengths` below grew by about 165 MB.
//
// Now a temporary is released when no resume point lies between it and the release, which goes
// in front of the first yield after it. Three cases stay leaking, as before:
//  - a value that goes out with the yield (`yield "a" + i`): the `{ value, done }` record holds
//    it without a count of its own, and the caller reads it after the call returns;
//  - a value made in front of a statement that has a yield inside it (`branchYield`): resuming
//    there reaches the end of the block without having made it;
//  - a value used after a yield.
//
// test-rc-generator-temporaries reads the IR for which is which.

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

    let branchTotal = 0;
    for (const v of branchYield()) branchTotal += v;
    assert(branchTotal == 6, "branchYield");

    let segments = 0;
    for (const v of twoSegments()) segments += v;
    assert(segments == 6, "twoSegments");

    print("done.");
}
