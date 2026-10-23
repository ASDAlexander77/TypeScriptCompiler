// A value stored into a local that owns nothing - one declared with no initializer, or in a
// finally or catch clause - stays alive while the local holds it. Such a local takes no count for
// what it is given, and under rc the value's own reference was given back at the end of the block
// that made it: `let n: N; if (c) { n = mk(); } use(n)`, or a loop in a finally assigning a local
// declared outside the loop, then read a freed block. Each read follows allocations of the same
// size, so a freed block is reused.
class V {
    tag = "";
    constructor(public x: number) {
        this.tag = "v" + x;
    }
}

class N {
    v: V;
    constructor(x: number) {
        this.v = new V(x);
    }
}

interface IX {
    v: V;
}

function mk(x: number) {
    return new N(x);
}

function churn() {
    const junk: N[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new N(999));
    }
}

function noInitializer(c: boolean) {
    let n: N;
    if (c) {
        n = mk(5);
    } else {
        n = mk(6);
    }

    churn();
    return n.v.x == (c ? 5 : 6) && n.v.tag == (c ? "v5" : "v6");
}

function noInitializerInLoop() {
    let n: N;
    n = mk(0);
    for (let i = 1; i < 4; i++) {
        n = mk(i);
    }

    churn();
    return n.v.x == 3;
}

function loopInFinally(ix: IX) {
    try {
    } finally {
        let n = mk(5);
        for (let j = 0; j < 3; j++) {
            ix.v = n.v;
            n = mk(j + 10);
            churn();
        }

        assert(n.v.x == 12 && n.v.tag == "v12", "a local of a finally, assigned in a loop");
    }

    churn();
    return ix.v.x == 11;
}

function ifInCatch() {
    let r: number = 0;
    try {
        throw 1;
    } catch (e) {
        let n = mk(1);
        if (r == 0) {
            n = mk(7);
        }

        churn();
        r = n.v.x;
    }

    return r == 7;
}

function main() {
    for (let i = 0; i < 3; i++) {
        assert(noInitializer(i % 2 == 0), "a local with no initializer, assigned in an if");
        assert(noInitializerInLoop(), "a local with no initializer, assigned in a loop");
        assert(loopInFinally({ v: new V(0) }), "a local of a finally");
        assert(ifInCatch(), "a local of a catch clause, assigned in an if");
    }

    print("done.");
}
