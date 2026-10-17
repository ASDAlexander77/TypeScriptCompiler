// The `using`s of one scope are disposed in reverse declaration order, as JS disposes them: the
// last one declared goes first, so a later resource may still use an earlier one while it is
// disposed. It used to be declaration order - `{ using a; using b; }` disposed `a` then `b`.
//
// The unwind cleanup region is generated from the same list, so a throw disposes in the same
// order as falling off the end, a `return`, a `break` or a `continue`.

let trail = "";

class Res {
    constructor(public name: string) {}

    [Symbol.dispose]() {
        trail += this.name + ";";
    }
}

function plainBlock() {
    {
        using a = new Res("a");
        using b = new Res("b");
        using c = new Res("c");
    }

    trail += "after;";
}

function functionBody() {
    using a = new Res("a");
    using b = new Res("b");
    trail += "body;";
}

function earlyReturn(f: boolean) {
    using a = new Res("a");
    {
        using b = new Res("b");
        using c = new Res("c");
        if (f) {
            return;
        }
    }

    trail += "after;";
}

function throwsInBlock() {
    {
        using a = new Res("a");
        using b = new Res("b");
        using c = new Res("c");
        throw 1;
    }
}

function throwsInTryBody() {
    try {
        using a = new Res("a");
        using b = new Res("b");
        throw 1;
    }
    catch (e: TypeOf<1>) {
        trail += "caught;";
    }
}

class Throws {
    constructor(public name: string) {}

    [Symbol.dispose]() {
        trail += this.name + ";";
        throw 1;
    }
}

// `b` is disposed first and throws; `a` is still disposed, by the cleanup the throw unwinds to
function throwingDispose() {
    try {
        using a = new Res("a");
        using b = new Throws("b");
        trail += "body;";
    }
    catch (e: TypeOf<1>) {
        trail += "caught;";
    }
}

function loopBody() {
    for (let i = 0; i < 2; i++) {
        using a = new Res("a" + i);
        using b = new Res("b" + i);
        if (i == 0) {
            continue;
        }

        break;
    }
}

function switchCase(n: number) {
    switch (n) {
        case 1:
            using a = new Res("a");
            using b = new Res("b");
            break;
    }

    trail += "after;";
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
    print(what + ": " + trail);
    assert(trail == expected, what);
}

function main() {
    check(plainBlock, "c;b;a;after;", "plain block");
    check(functionBody, "body;b;a;", "function body");
    check(() => earlyReturn(true), "c;b;a;", "return out of a nested block");
    check(() => earlyReturn(false), "c;b;after;a;", "falling off a nested block");
    check(() => assert(expectThrow(throwsInBlock), "must throw"), "c;b;a;", "throw out of a block");
    check(throwsInTryBody, "b;a;caught;", "throw out of a try body");
    check(throwingDispose, "body;b;a;caught;", "a dispose that throws");
    check(loopBody, "b0;a0;b1;a1;", "continue and break out of a loop body");
    check(() => switchCase(1), "b;a;after;", "break out of a switch case");

    print("done.");
}
