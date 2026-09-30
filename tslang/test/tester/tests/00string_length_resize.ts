// `s.length = n` resizes a string, and a string is a value: the resized one is a new string, and
// whoever held the old one still holds it. It used to reallocate the block in place, which frees
// the block when it moves - under every model but gc - while something else still held it. Under
// rc that was fatal whenever the string came in through a parameter: the default library's
// `"".clone().resize(n)`, behind padStart, padEnd, repeat, the regexp results and more, resized
// the caller's temporary through `this`, and the caller then released the freed block (heap
// corruption, 0xC0000374 on Windows).
//
// The block has to move for the bug to show, so every case grows its string well past the size
// it started at, and the heap is churned before anything is read back.

namespace __String {
    function copyOf(this: string): string {
        return this + "";
    }

    // the default library's `resize`
    function resized(this: string, newSize: index): string {
        this.length = newSize + 1;
        this[newSize] = <char>0;
        return this;
    }
}

class Vec {
    x: number;

    constructor(x: number) {
        this.x = x;
    }
}

class Box {
    text: string;

    constructor(text: string) {
        this.text = text;
    }
}

// Allocate over whatever has just been freed, so a use-after-free reads something else.
function churn() {
    for (let i = 0; i < 64; i++) {
        let filler = new Vec(999);
    }
}

function repeated(c: string, n: int): string {
    let s = "";
    for (let i = 0; i < n; i++) {
        s += c;
    }

    return s;
}

// The default library's shape: the string is the caller's temporary, borrowed by `this`.
function filled(n: int, c: string): string {
    const s = "ab".copyOf().resized(n);
    for (let i = 0; i < n; i++) {
        s[i] = c[0];
    }

    return s;
}

function resizedThroughParameter() {
    let last = "";
    for (let k = 0; k < 200; k++) {
        last = filled(100 + k, "x");
    }

    churn();
    return last == repeated("x", 299);
}

// A second variable holds the same string; resizing one of them leaves the other as it was.
function aliasKeepsItsString() {
    let a = "ab" + 1;
    let b = a;
    b.length = 64;
    b[0] = "z"[0];
    churn();

    return a == "ab1" && b == "zb1";
}

// A field is resized; a local still holding what the field held keeps it.
function fieldResized() {
    let box = new Box("cd" + 2);
    let before = box.text;
    box.text.length = 128;
    box.text[1] = "q"[0];
    churn();

    return before == "cd2" && box.text == "cq2";
}

// Smaller: the copy keeps only what fits, and `resized` writes the terminator.
function shrunk() {
    let s = "abcdef".copyOf().resized(2);
    churn();

    return s == "ab";
}

function main() {
    assert(resizedThroughParameter(), "a string resized through `this` leaves the caller's own alone");
    assert(aliasKeepsItsString(), "resizing a string leaves another variable's copy as it was");
    assert(fieldResized(), "resizing a field's string leaves a local's copy as it was");
    assert(shrunk(), "a string resized smaller keeps its start");

    print("done.");
}
