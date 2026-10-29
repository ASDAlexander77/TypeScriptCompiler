// An exception a call in a catch throws, taken by a typed catch of the same function: its catch variable reads what
// was thrown. A print of several values in the function was a variable-size alloca, which on Windows moves the stack
// of a function with exception funclets, and the CRT wrote the outer catch's object at the wrong address - a crash,
// or a stack overflow.
class Fail {
    constructor(public where: string) {}
}

function throwNumber(n: number) { throw n; }
function throwFail(where: string) { throw new Fail(where); }
function throwString(s: string) { throw s; }

function main() {
    let n = 0.0;
    try {
        try { throw 1; } catch (e: number) { print("inner", e); throwNumber(2.5); }
    } catch (e: number) {
        n = e;
    }
    assert(n == 2.5, "a number");

    let where = "";
    try {
        try { throwFail("first"); } catch (e: Fail) { print("inner", e.where); throwFail("second"); }
    } catch (e: Fail) {
        where = e.where;
    }
    assert(where == "second", "a class");

    let s = "";
    try {
        try { throw "a"; } catch (e: string) { throwString("b"); }
    } catch (e: string) {
        print("outer", e);
        s = e;
    }
    assert(s == "b", "a string");

    // through a finally on the way out
    let ran = false;
    n = 0;
    try {
        try { throw 1; } catch (e: number) { throwNumber(3); } finally { ran = true; }
    } catch (e: number) {
        n = e;
    }
    assert(ran && n == 3, "finally");

    print("done.");
}
