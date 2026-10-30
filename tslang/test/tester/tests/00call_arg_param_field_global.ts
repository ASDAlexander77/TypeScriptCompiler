// An argument read out of a field of an object a parameter names - `g(p.n)` - is borrowed by the
// callee, and the caller took no count for it. When the parameter is a global's object (`f(HG)`) and
// the callee gives that global's field a new value (`HG.n = mk(5)`), the store freed the block the
// argument read, under rc: `g` read freed memory, and a result that was the argument was freed too
// (mm_fuzz seed 71210). An argument read out of a global itself was already held across such a call
// (#647); now one read through a parameter's object is held as well, when the callee may give a
// global a new value. Each value is read after churn() has reused anything freed too early.
class V {
    constructor(public x: number) {}
}

class N {
    items: number[] = [1];
    constructor(public v: V) {}
}

class H {
    n: N = new N(new V(3));
}

function mk(i: number): N {
    return new N(new V(i));
}

function churn() {
    const hold: N[] = [];
    for (let i = 0; i < 64; i++) {
        hold.push(new N(new V(777)));
    }
}

const HG = new H();

// the callee reads its parameter after giving the global's field a new value
function readsAfter(a: N): number {
    HG.n = mk(5);
    churn();
    return a.v.x;
}

function callReadsAfter(p: H): number {
    return readsAfter(p.n);
}

// the callee gives back the argument it was given
function givesBack(a: N): N {
    HG.n = mk(6);
    return a;
}

function callGivesBack(p: H): number {
    const r = givesBack(p.n);
    churn();
    return r.v.x;
}

// ...into the very field it was read from
function storesBack(p: H): number {
    p.n = givesBack(p.n);
    churn();
    return p.n.v.x;
}

// through an element
class A {
    ns: N[] = [mk(8)];
}

const AG = new A();

function setsElement(a: N): number {
    AG.ns[0] = mk(9);
    churn();
    return a.v.x;
}

function callSetsElement(p: A): number {
    return setsElement(p.ns[0]);
}

function main() {
    assert(callReadsAfter(HG) == 3, "the callee reads the argument after the store");
    assert(HG.n.v.x == 5, "the store took");

    HG.n = mk(4);
    assert(callGivesBack(HG) == 4, "the callee gives the argument back");

    HG.n = mk(7);
    assert(storesBack(HG) == 7, "the argument stored back into its field");

    assert(callSetsElement(AG) == 8, "an element read through a parameter");
    assert(AG.ns[0].v.x == 9, "the element store took");

    print("done.");
}
