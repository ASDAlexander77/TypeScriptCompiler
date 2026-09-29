// A `const` whose initializer carries a reference - `const a = new C()`, `const s = "a" + k`,
// `const r = make()` - has no storage of its own: the name is folded into the value itself. So
// every later mention of the name is the very value that carried the reference, and a receiver
// that asks "does this carry a reference I can take over?" gets yes at every mention, not just
// the first. Taking it over is only right for a value that is used once, right there.
//
// Each case below had a receiver take over a reference the name still needed:
//   - a `let` in an inner scope took it, released it at its scope exit, and the name went on
//     reading freed memory;
//   - two receivers both took the one reference, so the value had two holders and one count;
//   - a receiver inside a loop took it on every iteration, from a value made once.
// And one without a name: `let b = h.c = new C()` hands the same `new` to the field and then to
// the local, and both took it.
//
// Every case calls `churn()` between the suspect release and the read, so that a freed block is
// claimed by something else and a use-after-free reads a wrong answer instead of a lucky one.
// See 00owned_transfer.ts for why that matters.

class C {
    x: number;
    v: number[] = [];

    constructor(x: number) {
        this.x = x;
    }
}

class H {
    c: C = new C(0);
    s: string = "";
}

function churn() {
    const keep: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C(999);
        c.v.push(99);
        c.v.push(99);
        c.v.push(99);
        c.v.push(99);
        c.v.push(99);
        c.v.push(99);
        c.v.push(99);
        keep.push(c);
    }
}

function churnStrings() {
    const keep: string[] = [];
    for (let i = 0; i < 64; i++) {
        keep.push("b" + (i % 10));
    }
}

function filled(n: number) {
    const c = new C(n);
    for (let i = 0; i < n; i++) {
        c.v.push(i);
    }

    return c;
}

// a `let` in an inner scope, then the name again after that scope has ended
function innerLetThenName() {
    const a = new C(3);
    a.v.push(1);
    a.v.push(2);
    a.v.push(3);
    {
        let b = a;
        b.x = 4;
    }

    churn();
    return a.v.length + a.x;
}

// the same `let` inside a loop: taken over on every iteration
function innerLetInLoop() {
    const a = new C(0);
    for (let i = 0; i < 5; i++) {
        a.v.push(i);
    }

    for (let i = 0; i < 3; i++) {
        let b = a;
        b.x += 1;
    }

    churn();
    return a.v.length + a.x;
}

// two `let`s of one name
function twoLets() {
    const a = new C(3);
    a.v.push(1);
    a.v.push(2);
    a.v.push(3);
    let r = 0;
    {
        let b = a;
        let c = a;
        r = b.v.length + c.v.length;
    }

    churn();
    return r + a.v.length;
}

// a push, then a `let` of the same name
function pushThenLet() {
    const arr: C[] = [];
    const c = new C(3);
    c.v.push(1);
    c.v.push(2);
    c.v.push(3);
    arr.push(c);
    {
        let d = c;
        d.x = 4;
    }

    churn();
    return arr[0].v.length + arr[0].x;
}

// a push inside a loop: one value, three elements
function pushInLoop() {
    let keep = new C(0);
    {
        const arr: C[] = [];
        const c = new C(5);
        for (let i = 0; i < 3; i++) {
            arr.push(c);
        }

        keep = arr[0];
    }

    churn();
    return keep.x;
}

// a fresh string stored into a field inside a loop: each overwrite released the one before it,
// and the one before it was the same string
function fieldStoreInLoop() {
    let k: number = 3;
    const h = new H();
    const s = "a" + k;
    for (let i = 0; i < 3; i++) {
        h.s = s;
    }

    churnStrings();
    return h.s == "a3" && s == "a3" ? 1 : 0;
}

// a call's result, named, then a `let` in an inner scope
function callResultThenName() {
    const a = filled(3);
    {
        let b = a;
        b.x = 4;
    }

    churn();
    return a.v.length + a.x;
}

// no name at all: an assignment's value handed on to a second receiver
function chainedAssignment() {
    const h = new H();
    {
        let b = h.c = new C(3);
        b.v.push(1);
        b.v.push(2);
        b.v.push(3);
    }

    churn();
    return h.c.v.length + h.c.x;
}

function main() {
    assert(innerLetThenName() == 7, "a name outlives an inner `let` of it");
    assert(innerLetInLoop() == 8, "a name outlives a `let` of it in a loop");
    assert(twoLets() == 9, "two `let`s of one name are two holders");
    assert(pushThenLet() == 7, "a pushed name outlives a `let` of it");
    assert(pushInLoop() == 5, "a name pushed in a loop is three references");
    assert(fieldStoreInLoop() == 1, "a string stored into a field in a loop survives the overwrites");
    assert(callResultThenName() == 7, "a named call result outlives an inner `let` of it");
    assert(chainedAssignment() == 6, "a field and a local assigned one `new` are two holders");

    print("done.");
}
