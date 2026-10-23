// A tuple or record of owned values as a value: an array literal typed as a tuple of objects
// (`[mk(1), mk(2)]` as `[N, N]`), built in a local or returned, and a returned record read through a
// `const` or destructured into `let`s. Under own the tuple literal took a second reference of each
// value (`ts.CreateTuple` beside the temporaries' releases), and a field taken out of a `const` record
// moved it, so a second read was a use after the move. Each read follows allocations of the same
// size, so a freed block would be reused.
class N {
    tag = "";
    constructor(public x: number) {
        this.tag = "n" + x;
    }
}

function churn() {
    const keep: N[] = [];
    for (let i = 0; i < 1000; i++) keep.push(new N(777));
    return keep.length;
}

function mk(i: number) {
    return new N(i);
}

function pair(): [N, N] {
    return [mk(1), mk(2)];
}

function mixed(): [N, number] {
    return [mk(3), 4];
}

function rec() {
    return { a: mk(5), b: mk(6) };
}

function check(n: N, x: number, what: string) {
    assert(n.x == x && n.tag == "n" + x, what);
}

function main() {
    const local: [N, N] = [mk(7), mk(8)];
    churn();
    check(local[0], 7, "a local tuple's first");
    check(local[1], 8, "a local tuple's second");

    const p = pair();
    churn();
    check(p[0], 1, "a returned tuple's first");
    check(p[1], 2, "a returned tuple's second");

    const m = mixed();
    churn();
    check(m[0], 3, "a mixed tuple's object");
    assert(m[1] == 4, "a mixed tuple's number");

    const r = rec();
    churn();
    check(r.a, 5, "a const record's first field");
    check(r.b, 6, "a const record's second field");
    check(r.a, 5, "a const record's first field, read again");

    let [a, b] = pair();
    churn();
    check(a, 1, "destructured first");
    check(b, 2, "destructured second");

    let sum = 0;
    for (let i = 0; i < 3; i++) {
        const q = pair();
        churn();
        sum += q[0].x + q[1].x;
    }

    assert(sum == 9, "a tuple per iteration");
    print("done.");
}
