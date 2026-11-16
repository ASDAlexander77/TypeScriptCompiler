// A local declared in a catch or finally clause holds what it is given: it keeps that object alive
// when the place it came from is given a new value. Under rc such a local took no count, so
// reassigning the global freed the object it still read (a fuzzer find, seed 920443).
class V {
    constructor(public x: number) {}
}

class N {
    items: V[];
    constructor(v: V) {
        this.items = [new V(v.x)];
    }
}

interface IY {
    items: V[];
}

@noinline
function mk(i: number): N {
    return new N(new V(i));
}

function churn() {
    const hold: N[] = [];
    for (let i = 0; i < 64; i++) {
        hold.push(new N(new V(777)));
    }
}

let G: N = mk(1);

function inFinally(): number {
    try {
    } finally {
        const k: IY = G;
        G = mk(5);
        churn();
        return k.items[0].x;
    }
}

function inCatch(): number {
    try {
        throw 1;
    } catch (e) {
        const k: N = G;
        G = mk(6);
        churn();
        return k.items[0].x;
    }
}

function main() {
    for (let i = 0; i < 100; i++) {
        G = mk(1);
        assert(inFinally() == 1, "finally");
        G = mk(2);
        assert(inCatch() == 2, "catch");
    }

    print("done.");
}
