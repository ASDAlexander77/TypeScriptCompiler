// A labelled `break` or `continue` owes every scope between itself and the statement it names:
// the disposals a `using` declared, and the references those scopes' locals took.
//
// The walk outwards used to forget the label after its first step, so a labelled jump stopped at
// the first loop body it met: `continue outer` written in an `if` in an inner loop never disposed
// the outer body's `using`. A jump to a labelled block matched nothing at all and ran on to the
// function's scope: it disposed the function's `using` there and then, and under rc released the
// function's locals - the string built in the block was freed before it was returned.
//
// An unlabelled `break` in a labelled block left only the block, not the loop around it, and a
// loop body written without braces had no scope for a jump to stop at.

let trail = "";

class Res {
    constructor(public name: string) {}

    [Symbol.dispose]() {
        trail += this.name + ";";
    }
}

function continueOuterFromIf() {
    outer: for (let i = 0; i < 2; i++) {
        using a = new Res("outer" + i);
        for (let j = 0; j < 2; j++) {
            using b = new Res("inner" + j);
            if (j == 0) {
                continue outer;
            }
        }
    }
}

function breakOuterFromIf() {
    outer: while (true) {
        using a = new Res("outer");
        while (true) {
            using b = new Res("inner");
            if (true) {
                break outer;
            }
        }
    }
}

function breakLabelledBlock() {
    using f = new Res("fn");
    a: {
        using x = new Res("blk");
        if (x.name == "blk") break a;
        trail += "unreached;";
    }

    trail += "after;";
}

function continueLabelledBlock() {
    let n = 0;
    b: {
        using y = new Res("blk" + n);
        n++;
        if (n < 3) continue b;
    }

    trail += "n" + n + ";";
}

function unlabelledBreakInLabelledBlock() {
    for (let i = 0; i < 2; i++) {
        using a = new Res("body" + i);
        l: {
            using c = new Res("blk" + i);
            if (i == 0) break;
            trail += "unreached;";
        }

        trail += "unreached;";
    }

    trail += "after;";
}

function bracelessBody() {
    using f = new Res("fn");
    for (let i = 0; i < 2; i++) if (i == 0) continue;
    let i = 0;
    while (i++ < 3) if (i == 2) break;
    trail += "after;";
}

function stringInBreakBlock() {
    let s = "";
    a: {
        s += "x";
        break a;
    }

    return s;
}

function stringInContinueBlock() {
    let s = "";
    a: b: {
        s += "x";
        if (s.length < 3) continue a;
    }

    return s;
}

function stringInLabelledIf(c: boolean) {
    let s = "y";
    a: if (c) {
        s += "x";
        break a;
    }

    return s;
}

function check(f: () => void, expected: string, what: string) {
    trail = "";
    f();
    print(what + ": " + trail);
    assert(trail == expected, what);
}

function main() {
    check(continueOuterFromIf, "inner0;outer0;inner0;outer1;", "continue outer from an if");
    check(breakOuterFromIf, "inner;outer;", "break outer from an if");
    check(breakLabelledBlock, "blk;after;fn;", "break a labelled block");
    check(continueLabelledBlock, "blk0;blk1;blk2;n3;", "continue a labelled block");
    check(unlabelledBreakInLabelledBlock, "blk0;body0;after;", "unlabelled break in a labelled block");
    check(bracelessBody, "after;fn;", "loop bodies without braces");

    assert(stringInBreakBlock() == "x", "string, break a block");
    assert(stringInContinueBlock() == "xxx", "string, continue a block");
    assert(stringInLabelledIf(true) == "yx", "string, break a labelled if");

    print("done.");
}
