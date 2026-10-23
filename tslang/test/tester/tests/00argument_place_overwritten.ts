// An argument read out of a local, a parameter or a field stays alive when the call overwrites
// where it was read: a later argument assigns it (`check(n, (n = mk(2), n), 1)`), or the callee
// overwrites a field or an element of an object it is given together with what was read out of
// it (`overwrite(n, n.items[0])`). The argument is borrowed, so under rc the overwrite freed the
// block the parameter still read. Each read follows allocations of the same size, so a freed block
// is reused.
class V {
    tag = "";
    constructor(public x: number) {
        this.tag = "v" + x;
    }
}

class N {
    v: V;
    items: V[];
    constructor(x: number) {
        this.v = new V(x);
        this.items = [new V(x + 100)];
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

function check(n: N, other: N, x: number) {
    churn();
    return n.v.x == x && n.v.tag == "v" + x && other.v.x > 0;
}

function localReassigned() {
    let n = mk(1);
    return check(n, (n = mk(2), n), 1) && n.v.x == 2;
}

function localReassignedInLoop() {
    let n = mk(0);
    for (let i = 1; i < 4; i++) {
        if (!check(n, (n = mk(i), n), i - 1)) {
            return false;
        }
    }

    return n.v.x == 3;
}

function parameterReassigned(p: N) {
    return check(p, (p = mk(7), p), 6);
}

class H {
    n = mk(0);
}

function fieldReassigned() {
    const h = new H();
    h.n = mk(3);
    return check(h.n, (h.n = mk(4), h.n), 3) && h.n.v.x == 4;
}

function overwriteElement(n: N, item: V) {
    n.items[0] = new V(5);
    churn();
    return item.x == 108 && item.tag == "v108";
}

function overwriteField(n: N, v: V) {
    n.v = new V(5);
    churn();
    return v.x == 9 && v.tag == "v9";
}

function main() {
    for (let i = 0; i < 3; i++) {
        assert(localReassigned(), "a local given before an argument that assigns it");
        assert(localReassignedInLoop(), "a local given before an argument that assigns it, in a loop");
        assert(parameterReassigned(mk(6)), "a parameter given before an argument that assigns it");
        assert(fieldReassigned(), "a local's field given before an argument that assigns it");

        let n = mk(8);
        assert(overwriteElement(n, n.items[0]), "an element given with its object, overwritten through it");
        n = mk(9);
        assert(overwriteField(n, n.v), "a field given with its object, overwritten through it");
    }

    print("done.");
}
