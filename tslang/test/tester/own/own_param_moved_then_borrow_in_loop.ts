// -mm=own: a parameter moved out (into an array, or to a call that keeps it), then given a borrow in
// a loop (fuzz 972189, 973334): the slot holds nothing it owns after the loop, but the borrow was
// released at the end of each turn, and its owner freed it again.
class V {
    x: number;
    a: number[];
    constructor(x: number) {
        this.x = x;
        this.a = [x];
    }
}

class H {
    v: V = new V(5);
}

const HG = new H();

@noinline
function mv(i: number): V {
    return new V(i);
}

@noinline
function keep(p: V, q: V): number {
    p = mv(8);
    return q.x;
}

function churn() {
    const j: V[] = [];
    for (let i = 0; i < 64; i++) j.push(mv(999));
}

@noinline
function intoArray(p: V): number {
    const a: V[] = [mv(1), p];
    let seen: number = 0;
    for (let j = 0; j < 2; j++) {
        p = a[0];
        seen = p.x;
    }

    return seen + a[1].x;
}

@noinline
function toCall(p: V): number {
    let sum: number = keep(p, HG.v);
    for (let j = 0; j < 2; j++) {
        p = HG.v;
        sum += p.x;
    }

    return sum;
}

function main() {
    for (let i = 0; i < 300; i++) {
        assert(intoArray(mv(i)) == 1 + i, "into array");
        churn();
        assert(toCall(mv(i)) == 15, "to call");
        churn();
        assert(HG.v.x == 5, "global");
    }

    print("done.");
}
