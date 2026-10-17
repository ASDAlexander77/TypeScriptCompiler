// A `return`, `break` or `continue` that leaves a `try` with a `finally` runs the finally first, and
// the finally can read every scope around the `try`. So the jump disposes and releases only the
// scopes inside the `try`; the copy of the finally that the jump runs disposes and releases the
// scopes from the `try` out to where the jump goes once it is done.
//
// A `return` used to dispose the `using`s around the `try` before the finally ran - the finally
// read a disposed resource, and JS disposes it after the finally. A `break` or `continue` used to
// dispose and release the scopes around the `try` up to its loop before the finally ran, so under
// -mm=rc the finally read freed strings.
//
// The values are read again after churn() wherever a release one too many would free them early.

let trail = "";

class Res {
    constructor(public name: string) {}

    [Symbol.dispose]() {
        churn();
        trail += "d:" + this.name + ";";
    }
}

function h() {
    throw 1;
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

// --- return ---

// a `using` in a block around the `try` is disposed after the finally
function returnUsingInBlock(pre: string) {
    {
        using r = new Res(pre + "1");
        try {
            return r.name + "r";
        }
        finally {
            churn();
            trail += "f:" + r.name + ";";
        }
    }
}

// the function's own `using`, and a return from the catch clause
function returnUsingFromCatch(pre: string) {
    using r = new Res(pre + "2");
    try {
        h();
    }
    catch {
        return r.name + "c";
    }
    finally {
        churn();
        trail += "f:" + r.name + ";";
    }

    return "";
}

// through two finallys: each `using` goes after the finally that reads it
function returnUsingThroughTwoFinallys(pre: string) {
    using a = new Res(pre + "3a");
    try {
        using b = new Res(pre + "3b");
        try {
            return b.name + "r";
        }
        finally {
            churn();
            trail += "i:" + a.name + b.name + ";";
        }
    }
    finally {
        churn();
        trail += "o:" + a.name + ";";
    }
}

// --- break / continue ---

// the `try` sits in a block nested in the loop body, and the finally reads the block's local
function breakFromNestedBlock(pre: string) {
    let out = "";
    for (let i = 0; i < 3; i++) {
        let t = pre + "4" + i;
        {
            let s = t + "s";
            try {
                if (i == 1) {
                    break;
                }
            }
            finally {
                churn();
                trail += s + ";";
            }

            out += s + ",";
        }
    }

    churn();
    return out;
}

function continueFromNestedBlock(pre: string) {
    let out = "";
    for (let i = 0; i < 3; i++) {
        let t = pre + "5" + i;
        {
            let s = t + "s";
            try {
                if (i == 1) {
                    continue;
                }
            }
            finally {
                churn();
                trail += s + t + ";";
            }

            out += s + ",";
        }
    }

    churn();
    return out;
}

// the `try` sits right in the loop body, whose local the finally reads
function breakFromLoopBody(pre: string) {
    let out = "";
    for (let i = 0; i < 3; i++) {
        let t = pre + "6" + i;
        try {
            if (i == 1) {
                break;
            }
        }
        finally {
            churn();
            trail += t + ";";
        }

        out += t + ",";
    }

    churn();
    return out;
}

// a `using` in the block is disposed after the finally, both for a `break` and a `continue`
function jumpUsingInBlock(pre: string) {
    for (let i = 0; i < 3; i++) {
        {
            using r = new Res(pre + "7" + i);
            try {
                if (i == 0) {
                    continue;
                }

                if (i == 1) {
                    break;
                }
            }
            finally {
                churn();
                trail += "f:" + r.name + ";";
            }
        }
    }
}

// a labelled `continue` leaves the inner loop too: the inner body's local and the outer body's go
// after the finally
function labelledContinue(pre: string) {
    let out = "";
    outer: for (let i = 0; i < 2; i++) {
        let a = pre + "8" + i;
        for (let j = 0; j < 2; j++) {
            let b = a + j;
            {
                let s = b + "s";
                try {
                    if (j == 0) {
                        continue outer;
                    }
                }
                finally {
                    churn();
                    trail += a + "|" + s + ";";
                }
            }
        }

        out += "never;";
    }

    churn();
    return out;
}

// through two finallys: each releases what it reads only once it is done
function breakThroughTwoFinallys(pre: string) {
    for (let i = 0; i < 3; i++) {
        let a = pre + "9" + i;
        {
            let b = a + "b";
            try {
                let c = b + "c";
                {
                    let d = c + "d";
                    try {
                        if (i == 1) {
                            break;
                        }
                    }
                    finally {
                        churn();
                        trail += "i:" + c + "|" + d + ";";
                    }
                }
            }
            finally {
                churn();
                trail += "o:" + a + "|" + b + ";";
            }
        }
    }
}

// a `break` out of a case of a switch
function breakFromSwitchCase(pre: string, k: number) {
    switch (k) {
        case 1: {
            let s = pre + "b";
            {
                let u = s + "u";
                try {
                    break;
                }
                finally {
                    churn();
                    trail += s + u + ";";
                }
            }
        }
        default:
            trail += "default;";
    }
}

// a labelled block
function breakLabelledBlock(pre: string) {
    blk: {
        let s = pre + "c";
        {
            let u = s + "u";
            try {
                break blk;
            }
            finally {
                churn();
                trail += s + u + ";";
            }
        }
    }

    trail += "after;";
}

function main() {
    let pre = "p";

    trail = "";
    let r1 = returnUsingInBlock(pre);
    churn();
    print("return, using in a block: " + r1 + " " + trail);
    assert(r1 == "p1r" && trail == "f:p1;d:p1;", "return, using in a block");

    trail = "";
    let r2 = returnUsingFromCatch(pre);
    churn();
    print("return from a catch, using: " + r2 + " " + trail);
    assert(r2 == "p2c" && trail == "f:p2;d:p2;", "return from a catch, using");

    trail = "";
    let r3 = returnUsingThroughTwoFinallys(pre);
    churn();
    print("return through two finallys, using: " + r3 + " " + trail);
    assert(r3 == "p3br" && trail == "i:p3ap3b;d:p3b;o:p3a;d:p3a;", "return through two finallys, using");

    trail = "";
    let r4 = breakFromNestedBlock(pre);
    churn();
    print("break from a nested block: " + r4 + " " + trail);
    assert(r4 == "p40s," && trail == "p40s;p41s;", "break from a nested block");

    trail = "";
    let r5 = continueFromNestedBlock(pre);
    churn();
    print("continue from a nested block: " + r5 + " " + trail);
    assert(r5 == "p50s,p52s," && trail == "p50sp50;p51sp51;p52sp52;", "continue from a nested block");

    trail = "";
    let r6 = breakFromLoopBody(pre);
    churn();
    print("break from a loop body: " + r6 + " " + trail);
    assert(r6 == "p60," && trail == "p60;p61;", "break from a loop body");

    trail = "";
    jumpUsingInBlock(pre);
    churn();
    print("break and continue, using: " + trail);
    assert(trail == "f:p70;d:p70;f:p71;d:p71;", "break and continue, using");

    trail = "";
    let r8 = labelledContinue(pre);
    churn();
    print("labelled continue: " + r8 + " " + trail);
    assert(r8 == "" && trail == "p80|p800s;p81|p810s;", "labelled continue");

    trail = "";
    breakThroughTwoFinallys(pre);
    churn();
    print("break through two finallys: " + trail);
    assert(trail == "i:p90bc|p90bcd;o:p90|p90b;i:p91bc|p91bcd;o:p91|p91b;", "break through two finallys");

    trail = "";
    breakFromSwitchCase(pre, 1);
    churn();
    print("break from a switch case: " + trail);
    assert(trail == "pbpbu;", "break from a switch case");

    trail = "";
    breakLabelledBlock(pre);
    churn();
    print("break a labelled block: " + trail);
    assert(trail == "pcpcu;after;", "break a labelled block");

    print("done.");
}
