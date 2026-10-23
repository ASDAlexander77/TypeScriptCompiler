// A `const` made by `new` and then chosen by a conditional expression - `b ? h : h2` - is given
// back at the end of its block. The branch hands it on through `ts.Result`, a terminator, and that
// kept it from ever being released: under rc and own neither side was freed. What reads the chosen
// value retains it, so the release stays balanced; conditional-named-const-release.cmake checks the
// releases under rc. Here each value is read after allocations of the same size have reused
// anything freed too early.
class C {
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

function constChosen(b: boolean) {
    const h = new C(3);
    const h2 = new C(30);
    const each: C = b ? h : h2;
    churn();
    return each.dist + h.dist + h2.dist;
}

function letChosen(b: boolean) {
    const h = new C(3);
    const h2 = new C(30);
    let each = b ? h : h2;
    churn();
    return each.dist + h.dist + h2.dist;
}

function chosenOrNew(b: boolean) {
    const h = new C(3);
    const each: C = b ? h : new C(30);
    churn();
    return each.dist + h.dist;
}

function nestedChosen(a: boolean, b: boolean) {
    const h = new C(3);
    const h2 = new C(30);
    const each: C = a ? h : b ? h2 : h;
    churn();
    return each.dist + h.dist + h2.dist;
}

function chosenInLoop() {
    const h = new C(3);
    const h2 = new C(30);
    let sum = 0;
    for (let i = 0; i < 10; i++) {
        const each: C = i % 2 == 0 ? h : h2;
        sum += each.dist;
    }

    churn();
    return sum + h.dist + h2.dist;
}

function returnChosen(b: boolean) {
    const h = new C(3);
    const h2 = new C(30);
    return b ? h : h2;
}

// the local declared without a value owns nothing, so `h` is what keeps the value it is given
// alive after the `if`: it is not released at the end of the `if`
function assignedOutside(b: boolean) {
    const h2 = new C(30);
    let n: C;
    if (b) {
        const h = new C(3);
        n = b ? h : h2;
    } else {
        n = h2;
    }

    churn();
    return n.dist;
}

function main() {
    assert(constChosen(true) == 3 + 33 && constChosen(false) == 30 + 33, "a const chosen from two");
    assert(letChosen(true) == 3 + 33 && letChosen(false) == 30 + 33, "a let chosen from two");
    assert(chosenOrNew(true) == 3 + 3 && chosenOrNew(false) == 30 + 3, "one side made by the choice");
    assert(nestedChosen(false, true) == 30 + 33 && nestedChosen(false, false) == 3 + 33, "a choice inside a choice");
    assert(chosenInLoop() == 5 * 3 + 5 * 30 + 33, "a choice in a loop");

    const r = returnChosen(false);
    churn();
    assert(r.dist == 30, "the chosen one returned");

    assert(assignedOutside(true) == 3 && assignedOutside(false) == 30, "a choice given to a local outside");
    print("done.");
}
