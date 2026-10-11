// The elements of a boolean array literal read back wrong: the constant data was an LLVM vector of
// i1, which packs its elements into bits, while elements are read one byte apart. So `if (F[i])`
// was never taken, a count of the true elements came out as 10 for four elements, and F[1] was
// false.

const FLAGS = [false, true, false, true];

function countTrue(flags: boolean[]) {
    let n = 0;
    for (let i = 0; i < flags.length; i++) {
        if (flags[i]) {
            n++;
        }
    }

    return n;
}

function locals() {
    let a = [true, false, true];
    assert(a[0], "a[0]");
    assert(!a[1], "a[1]");
    assert(a[2], "a[2]");

    let n = 0;
    for (const x of a) {
        if (x) n++;
    }

    assert(n == 2, "for-of count");
}

function many() {
    // more elements than fit in a byte of bits
    const m = [true, false, false, true, true, false, true, false, true, true, false, true];
    assert(countTrue(m) == 7, "twelve elements");
    assert(m[11], "m[11]");
    assert(!m[10], "m[10]");
}

function main() {
    assert(FLAGS.length == 4, "length");
    assert(countTrue(FLAGS) == 2, "count");
    const b: boolean = FLAGS[1];
    assert(b, "FLAGS[1]");
    assert(!FLAGS[0], "FLAGS[0]");
    assert(FLAGS[3], "FLAGS[3]");

    locals();
    many();

    print("done.");
}
