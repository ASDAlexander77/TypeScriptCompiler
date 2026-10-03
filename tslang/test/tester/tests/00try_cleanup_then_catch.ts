// A try whose scope cleans up on the way out (an owning `let`, a `using`) hands an exception thrown
// in it to its catch (#457, #456). On Windows the cleanup went through a catch-all that rethrew,
// and the catch never saw the rethrow when the function also allocated on the stack dynamically (a
// two-argument `print`): the program terminated.
class Node {
    v = 0;
}

class Res {
    [Symbol.dispose]() {
        print("disposed");
    }
}

function fail(): Node {
    throw "x";
}

function letThenThrow(s: Node) {
    let caught = false;
    try {
        let t = s;
        print("in try", t.v);
        throw 1;
    } catch {
        caught = true;
    }

    return caught;
}

function printInCatch() {
    let caught = false;
    try {
        let t = new Node();
        throw 1;
    } catch {
        print("caught", 5);
        caught = true;
    }

    return caught;
}

function usingThenThrow() {
    let caught = false;
    try {
        using r = new Res();
        throw 1;
    } catch {
        print("caught", 6);
        caught = true;
    }

    return caught;
}

function initializerThrows() {
    let caught = false;
    try {
        let back = fail();
        print("not reached", back.v);
    } catch {
        caught = true;
    }

    return caught;
}

function castThrows() {
    const n: any = 5;
    let caught = false;
    try {
        let back = <Node>n;
        print("not reached", back.v);
    } catch {
        caught = true;
    }

    return caught;
}

function constControl(s: Node) {
    let caught = false;
    try {
        const t = s;
        print("in try", t.v);
        throw 1;
    } catch {
        caught = true;
    }

    return caught;
}

function main() {
    assert(letThenThrow(new Node()), "an owning let, a two-argument print, then a throw");
    assert(printInCatch(), "an owning let, then a two-argument print in the catch");
    assert(usingThenThrow(), "a using, then a two-argument print in the catch");
    assert(initializerThrows(), "a let whose initializer throws (#456)");
    assert(castThrows(), "a let whose cast throws (#456)");
    assert(constControl(new Node()), "a const needs no cleanup");

    for (let i = 0; i < 100; i++) {
        assert(letThenThrow(new Node()));
    }

    print("done.");
}
