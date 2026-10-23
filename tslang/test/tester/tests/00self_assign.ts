// A local given its own value - `n = n`, or `n = (n = mk(5), n)` - keeps it, and can be moved or read
// afterwards. Under own the store back read the value out of the local first, which counted as moving
// it, so the release of what the local held came after the move and the program was refused. Each
// read follows allocations of the same size, so a freed block would be reused.
class V {
    tag = "";
    constructor(public x: number) {
        this.tag = "v" + x;
    }
}

class Junk {
    a = -1;
    b = -2;
    tag = "junk";
}

function churn() {
    let keep: Junk[] = [];
    for (let i = 0; i < 1000; i++) keep.push(new Junk());
    return keep.length;
}

function mk(x: number) {
    return new V(x);
}

function take(v: V) {
    churn();
    return v.tag == "v" + v.x ? v.x : -1;
}

function main() {
    let s = 0;
    for (let i = 0; i < 3; i++) {
        let n = mk(i);
        n = n;
        const m = n;
        churn();
        s += take(m);
    }
    assert(s == 3, "n = n, then moved");

    let k = mk(1);
    k = (k = mk(5), k);
    churn();
    assert(take(k) == 5, "n = (n = mk(5), n)");

    print("done.");
}
