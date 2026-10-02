// -mm=own, phase 7: an object seen through an interface is that object's block (a view), so the
// interface value and the object have one owner between them. Each case reads through a churn()
// after the point where a second owner would have freed the block.
interface I {
    x: number;
    get(): number;
}

class C implements I {
    v: number[] = [1];
    constructor(public x: number) {}
    get() {
        return this.x + this.v.length;
    }
}

class Junk {
    v: number[] = [];
    constructor(public x: number) {}
}

function churn() {
    const junk: Junk[] = [];
    for (let i = 0; i < 64; i++) {
        const j = new Junk(999);
        j.v.push(99);
        junk.push(j);
    }
}

function show(i: I) {
    churn();
    return i.get();
}

// an object literal made straight into an interface
function make(k: number): I {
    return {
        x: k,
        get(): number {
            return this.x;
        },
    };
}

function main() {
    let t = 0;
    for (let i = 0; i < 20000; i++) {
        const n = i % 10;

        // a temporary class instance given as an interface
        t += show(new C(n));

        // an interface holding a class instance
        const ci: I = new C(n);
        churn();
        t += ci.get();

        // an object literal kept in a `let`, seen through an interface: the interface borrows it
        let raw = {
            x: n,
            get(): number {
                return this.x * 2;
            },
        };
        let ri = <I>raw;
        churn();
        t += ri.get() + raw.x;

        // a folded object literal given once as an interface
        const o = {
            x: n,
            get(): number {
                return this.x + 1;
            },
        };
        t += show(o);
        t += o.x;

        const m = make(n);
        churn();
        t += m.get();
    }

    print(t);
    assert(t == 780000, "t");
    print("done.");
}
