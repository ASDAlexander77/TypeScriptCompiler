// TSLANG 213: under -mm=rc a `for (let s ...)` of a string (or another reference) a closure captures has a binding of
// its own each iteration, as a number's has (each closure saw the loop's last string); one the body assigns gives its
// value back to the loop's. (-mm=own refuses the capture.)

class Box {
    constructor(public n: number) {}
}

function main() {
    const fs: (() => string)[] = [];
    for (let s = "a"; s.length < 4; s = s + "a") {
        fs.push(() => s);
    }

    assert(fs.length == 3);
    assert(fs[0]() == "a");
    assert(fs[1]() == "aa");
    assert(fs[2]() == "aaa");

    const gs: (() => string)[] = [];
    for (let s = "x" + gs.length; s.length < 7; s = s + "a") {
        s = s + "b";
        gs.push(() => s);
    }

    assert(gs.length == 3);
    assert(gs[0]() == "x0b");
    assert(gs[1]() == "x0bab");
    assert(gs[2]() == "x0babab");

    const hs: (() => number)[] = [];
    for (let b = new Box(0); b.n < 4; b = new Box(b.n + 1)) {
        if (b.n == 1) {
            b = new Box(b.n + 1);
            continue;
        }

        hs.push(() => b.n);
    }

    assert(hs.length == 2);
    assert(hs[0]() == 0);
    assert(hs[1]() == 3);

    let caught = "";
    const ks: (() => string)[] = [];
    try {
        for (let s = "t"; s.length < 10; s = s + "y") {
            ks.push(() => s);
            if (s.length > 2) {
                throw s;
            }
        }
    } catch (e) {
        caught = <string>e;
    }

    assert(caught == "tyy");
    assert(ks[0]() == "t" && ks[2]() == "tyy");

    print("done.");
}
