// -mm=own: a string parameter the body assigns is a local of its own, made from a copy of the
// caller's string, which owns each string assigned to it and is given back or moved out. Left as
// the parameter's slot, what was assigned had no owner: it leaked, and `return s` gave back a copy
// of it, read up to its first NUL. The default library's `"".clone().resize(n)` (padStart,
// repeat, the regexp results) so returned a one-byte string, which its caller filled with n
// characters: a heap overrun, and the regexp results came back corrupted.
//
// The block has to move for the bug to show, so the strings grow well past their size, and the
// heap is churned before anything is read back.

namespace __String {
    function copyOf(this: string, add = ""): string {
        return this + add;
    }

    // the default library's `resize`: `this.length = n` assigns `this` a new string
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
function filled(n: int): string {
    let s = "".copyOf().resized(n);
    for (let i = 0; i < n; i++) {
        s[i] = <char>65;
    }

    return s;
}

// `s.length = n` of a parameter named as one
function grown(s: string, n: int): string {
    s.length = n + 1;
    s[n] = <char>0;
    for (let i = 0; i < n; i++) {
        s[i] = <char>66;
    }

    return s;
}

function appended(s: string, t: string): string {
    s = s + t;
    return s;
}

function appendedInLoop(s: string, n: int): string {
    for (let i = 0; i < n; i++) {
        s += "x";
    }

    return s;
}

function appendedOnOnePath(s: string, add: boolean): string {
    if (add) {
        s = s + "!";
    }

    return s;
}

function main() {
    const a = filled(40);
    churn();
    const b = filled(40);
    churn();
    assert(a == repeated("A", 40));
    assert(b == a);

    const original = "cd";
    const g = grown(original, 40);
    churn();
    assert(g == repeated("B", 40));
    assert(original == "cd");

    const p = "ab";
    const q = appended(p, repeated("y", 40));
    churn();
    assert(q == "ab" + repeated("y", 40));
    assert(p == "ab");

    assert(appendedInLoop("z", 40) == "z" + repeated("x", 40));
    assert(appendedOnOnePath("hi", true) == "hi!");
    assert(appendedOnOnePath("hi", false) == "hi");

    print("done.");
}
