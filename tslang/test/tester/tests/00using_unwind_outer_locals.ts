// An exception through a block's `using` gives back what the scopes around the block took, too.
//
// A block with a `using` is wrapped in a catch-less TryOp whose cleanup region disposes and
// releases the block's own scope, then lets the exception keep unwinding. A scope around it with
// no cleanup region of its own - here, mostly, the function body - got no other chance, so
// `let s = a + b; { using r = new Res(); throw 1; }` leaked `s` under -mm=rc. The ownership
// verifier (--verify-ownership, swept over this corpus in ctest) reported every such shape; this
// file is among them, and it is quiet now.
//
// The walk outwards stops at the first scope that is itself a TryOp's body: that TryOp's own
// cleanup gives its scope back, and when it has a catch, everything further out is live after it.
// The `caughtFurtherOut` cases check that side - a release too many there frees what is still read.

let trail = "";

class Res {
    constructor(public name: string) {}

    [Symbol.dispose]() {
        trail += this.name + ";";
    }
}

// allocates over whatever was freed last, so that a string released one time too many no longer
// reads as itself
function churn() {
    let junk = "";
    for (let i = 0; i < 64; i++) {
        junk = junk + "#";
    }

    return junk.length;
}

function returnFromBlock(pre: string) {
    let s = pre + "1";
    {
        using a = new Res("a");
        return s;
    }
}

function throwFromBlock(pre: string) {
    let s = pre + "2";
    {
        using a = new Res("a");
        trail += s + ";";
        throw 1;
    }
}

function throwThroughPlainScope(pre: string) {
    let s = pre + "3";
    {
        let t = s + "t";
        {
            using a = new Res("a");
            trail += t + ";";
            throw 1;
        }
    }
}

function throwFromSwitchCase(pre: string, n: number) {
    let s = pre + "4";
    switch (n) {
        case 1:
            using a = new Res("a");
            trail += s + ";";
            throw 1;
    }

    return s;
}

function throwFromLoopBody(pre: string) {
    let s = pre + "5";
    for (let i = 0; i < 3; i++) {
        let t = s + i;
        {
            using a = new Res("a" + i);
            if (i == 1) {
                throw 1;
            }
        }
    }
}

function caughtFurtherOut(pre: string) {
    let s = pre + "6";
    try {
        let t = s + "t";
        {
            using a = new Res("a");
            throw 1;
        }
    }
    catch (e: TypeOf<1>) {
        trail += "caught;";
    }

    churn();
    return s;
}

function caughtFurtherOutSwitch(pre: string, n: number) {
    let s = pre + "7";
    try {
        switch (n) {
            case 1:
                using a = new Res("a");
                throw 1;
        }
    }
    catch (e: TypeOf<1>) {
        trail += "caught;";
    }

    churn();
    return s;
}

function expectThrow(f: () => void) {
    try {
        f();
    }
    catch (e: TypeOf<1>) {
        return true;
    }

    return false;
}

function check(f: () => void, expected: string, what: string) {
    trail = "";
    f();
    churn();
    print(what + ": " + trail);
    assert(trail == expected, what);
}

function main() {
    let pre = "p";

    trail = "";
    let r = returnFromBlock(pre);
    churn();
    print("return out of a block: " + r + " " + trail);
    assert(r == "p1" && trail == "a;", "return out of a block");

    check(() => assert(expectThrow(() => throwFromBlock(pre)), "must throw"), "p2;a;", "throw out of a block");
    check(() => assert(expectThrow(() => throwThroughPlainScope(pre)), "must throw"), "p3t;a;", "throw through a plain scope");
    check(() => assert(expectThrow(() => { throwFromSwitchCase(pre, 1); }), "must throw"), "p4;a;", "throw out of a switch case");
    check(() => assert(expectThrow(() => throwFromLoopBody(pre)), "must throw"), "a0;a1;", "throw out of a loop body");

    trail = "";
    let c = caughtFurtherOut(pre);
    print("caught further out: " + c + " " + trail);
    assert(c == "p6" && trail == "a;caught;", "caught further out");

    trail = "";
    let cs = caughtFurtherOutSwitch(pre, 1);
    print("caught further out, switch: " + cs + " " + trail);
    assert(cs == "p7" && trail == "a;caught;", "caught further out, switch");

    print("done.");
}
