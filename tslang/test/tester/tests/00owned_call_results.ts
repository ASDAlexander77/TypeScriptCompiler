// Every function retains its result on the way out, so a receiver that retains it again is one
// owner above the truth and the value is never freed. Deciding which callees actually do that
// cannot be settled where MLIRGen builds the call, so it is settled afterwards, by a pass that
// looks at each function's returns instead of predicting them (section 9.27).
//
// These guard the direction that corrupts. Under-consuming leaks, which is invisible from inside
// the program; over-consuming takes the value below its true owner count and frees it while a
// receiver still points at it. Every case therefore calls `churn()` between the last release and
// the read, so a freed block is claimed by something else and a use-after-free shows up as a
// wrong answer rather than as the value that used to be there.
//
// Excluded from consumption, and each verified to still keep its retain: a callee with no body
// (`declare`), an indirect call through a function value, and a generator. Those are the shapes
// where consuming would free a reference nobody took.
//
// See docs/reference-counting-evaluation.md section 9.27.

class Leaf {
    n: number;

    constructor(n: number) {
        this.n = n;
    }
}

class Holder {
    item: Leaf;
}

// Allocate over whatever has just been freed, so a use-after-free reads something else.
function churn() {
    for (let i = 0; i < 64; i++) {
        let filler = new Leaf(999);
    }
}

function make(v: number): Leaf {
    let x = new Leaf(v);
    return x;
}

// a call result held by a local, read after an inner scope has released its own reference
function callResultOutlivesInnerScope() {
    let out = new Leaf(0);
    {
        let tmp = make(5);
        out = tmp;
    }

    churn();
    return out.n;
}

// the plainest shape: consume, then use
function callResultIntoLocal() {
    let a = make(6);
    churn();

    return a.n;
}

// two locals holding one call result
function callResultShared() {
    let a = make(7);
    let b = a;
    churn();

    return a.n + b.n;
}

// a call result stored into a field, outliving the local that received it
function callResultIntoField() {
    let h = new Holder();
    {
        let tmp = make(8);
        h.item = tmp;
    }

    churn();
    return h.item.n;
}

// a call result captured by an array literal, and one pushed
function callResultIntoArray() {
    let arr = [make(3), make(4)];
    arr.push(make(2));
    churn();

    return arr[0].n + arr[1].n + arr[2].n;
}

// a call whose result is another call's result, so the transfer passes through two frames
function forward(v: number): Leaf {
    return make(v);
}

function callResultThroughTwoFrames() {
    let a = forward(9);
    churn();

    return a.n;
}

// a method returning a heap value
class Factory {
    build(v: number): Leaf {
        let x = new Leaf(v);
        return x;
    }
}

function methodResult() {
    let f = new Factory();
    let a = f.build(11);
    churn();

    return a.n;
}

// an arrow function's result - a concise body returns without going through the return
// statement, so it needed a retain of its own before it could qualify at all
function arrowResult() {
    const mk = (v: number): Leaf => new Leaf(v);
    let a = mk(12);
    churn();

    return a.n;
}

function main() {
    assert(callResultOutlivesInnerScope() == 5, "a consumed call result outlives the local that took it");
    assert(callResultIntoLocal() == 6, "the plain case survives");
    assert(callResultShared() == 14, "two locals holding one call result both stay valid");
    assert(callResultIntoField() == 8, "a call result stored into a field outlives the receiver");
    assert(callResultIntoArray() == 9, "call results captured by an array stay valid");
    assert(callResultThroughTwoFrames() == 9, "a result forwarded through two frames survives");
    assert(methodResult() == 11, "a method's result is consumed like a function's");
    assert(arrowResult() == 12, "an arrow function's result too");

    print("done.");
}
