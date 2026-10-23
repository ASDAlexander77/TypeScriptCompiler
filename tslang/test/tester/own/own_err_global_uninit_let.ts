// -mm=own rejects: `let l!: V` inside `f`'s loop stores nothing, so the slot keeps the value of
// the round before - the read of `G` stored in round 1 - and the later rounds read it after
// `G = new V(i + 1)` freed that block. A declaration without an initializer gives a holder no new
// value.
class V {
    a: number[] = [];
    constructor(public x: number) {
        for (let i = 0; i < 64; i++) {
            this.a.push(i);
        }
    }
}

@noinline
function churn() {
    const junk: V[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new V(999));
    }
}

let G = new V(0);

function f() {
    for (let i = 0; i < 4; i++) {
        let l!: V;
        if (i > 1) {
            print(l.x, l.a.length);
        }
        const c = G;
        if (i == 1) {
            l = c;
        }
        G = new V(i + 1);
        churn();
    }
}

function main() {
    f();
    print(G.x);
    print("done.");
}
