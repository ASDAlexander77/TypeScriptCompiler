// -mm=own: `c ? h : h2` bound inside a loop, of locals made before it, borrows whichever it
// chooses and moves neither - a `const` or a `let`, of a class or an interface, folded to a select
// or kept as a branch, and with one side made inside the loop. Both stay the function's after the
// loop.
interface HasDist {
    dist: number;
}

class C implements HasDist {
    v: number[] = [];
    constructor(public dist: number) {}
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C(999);
        c.v.push(99);
        junk.push(c);
    }
}

function constClass() {
    const h = new C(3);
    const h2 = new C(30);
    let sum: number = 0;
    for (let i = 0; i < 10; i++) {
        const each: C = i % 2 == 0 ? h : h2;
        sum += each.dist;
    }

    churn();
    return sum + h.dist + h2.dist;
}

function letClass() {
    const h = new C(3);
    const h2 = new C(30);
    let sum: number = 0;
    for (let i = 0; i < 10; i++) {
        let each: C = i % 2 == 0 ? h : h2;
        sum += each.dist + each.v.length;
    }

    churn();
    return sum + h.dist + h2.dist;
}

function constInterface() {
    const h: HasDist = new C(3);
    const h2: HasDist = new C(30);
    let sum: number = 0;
    for (let i = 0; i < 10; i++) {
        const each: HasDist = i % 2 == 0 ? h : h2;
        sum += each.dist;
    }

    churn();
    return sum + h.dist + h2.dist;
}

function letInterface() {
    const h: HasDist = new C(3);
    const h2: HasDist = new C(30);
    let sum: number = 0;
    for (let i = 0; i < 10; i++) {
        let each: HasDist = i % 2 == 0 ? h : h2;
        sum += each.dist;
    }

    churn();
    return sum + h.dist + h2.dist;
}

let calls: number = 0;
function odd(i: number) {
    calls++;
    return i % 2 != 0;
}

// a call on one side keeps the merge a branch
function branched() {
    const h = new C(3);
    const h2 = new C(30);
    let sum: number = 0;
    for (let i = 0; i < 10; i++) {
        const each: C = i % 4 == 0 ? h : (odd(i) ? h2 : h);
        sum += each.dist;
    }

    churn();
    return sum + h.dist + h2.dist;
}

function madeInside() {
    const h2: HasDist = new C(30);
    let sum: number = 0;
    for (let i = 0; i < 10; i++) {
        const h: HasDist = new C(i);
        const each = i % 2 == 0 ? h : h2;
        sum += each.dist;
    }

    churn();
    return sum + h2.dist;
}

function main() {
    assert(constClass() == 5 * 3 + 5 * 30 + 33, "a const of two classes");
    assert(letClass() == 5 * 3 + 5 * 30 + 33, "a let of two classes");
    assert(constInterface() == 5 * 3 + 5 * 30 + 33, "a const of two interfaces");
    assert(letInterface() == 5 * 3 + 5 * 30 + 33, "a let of two interfaces");
    assert(branched() == 5 * 3 + 5 * 30 + 33, "a merge kept as a branch");
    assert(calls == 7, "the call on one side");
    assert(madeInside() == (0 + 2 + 4 + 6 + 8) + 5 * 30 + 30, "one side made in the loop");
    print("done.");
}
