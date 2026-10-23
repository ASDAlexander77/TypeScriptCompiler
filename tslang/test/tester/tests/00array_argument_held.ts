// A new array given straight to a call (`g([mk(1)])`) stays alive while the callee takes a count
// of it and gives it back. Under rc a new array is born at count 0 and nothing else owned it: an
// alias in a loop or a try (`const a = p`), or the parameter assigned (copied into a local that
// owns it), freed the array the callee still read. Each read follows allocations of the same
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

function mk(x: number) {
    return new N(x);
}

function churn() {
    const junk: N[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new N(999));
    }
}

function aliasInLoop(p: N[]) {
    for (let i = 0; i < 3; i++) {
        const a: N[] = p;
        churn();
    }

    return p[0].v.x == 1 && p[1].v.tag == "v2";
}

function aliasInTry(p: N[]) {
    try {
        let a: N[] = p;
    } finally {
        p.push(mk(3));
    }

    churn();
    return p.length == 3 && p[2].v.x == 3 && p[0].v.x == 1;
}

function assigned(p: N[]) {
    const first = p[0].v.x;
    p = [mk(7)];
    churn();
    return first == 1 && p[0].v.x == 7;
}

function passedOn(p: N[]) {
    const ok = assigned(p);
    churn();
    return ok && p[0].v.x == 1 && p[0].v.tag == "v1";
}

function numbers(p: number[]) {
    for (let i = 0; i < 3; i++) {
        const a = p;
        churn();
    }

    return p[0] + p[1] + p[2];
}

function main() {
    for (let i = 0; i < 3; i++) {
        assert(aliasInLoop([mk(1), mk(2)]), "an alias of a new array, in a loop");
        assert(aliasInTry([mk(1), mk(2)]), "an alias of a new array, in a try");
        assert(assigned([mk(1)]), "a new array whose parameter is assigned");
        assert(passedOn([mk(1)]), "a new array given on to a call that assigns its parameter");
        assert(numbers([1, 2, 3]) == 6, "a constant array, copied for the call");
    }

    print("done.");
}
