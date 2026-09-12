// A closure inside a closure, both naming the same outer variable. The inner box owns the
// variable's cell exactly as the outer one does, so it has to take a reference to it; without
// that, closing the inner box gives back a count nobody added and the cell is freed while the
// outer closure - and the frame - still point at it.
//
// A freed cell keeps its contents until something else is allocated over it, so every case here
// builds in one place, allocates hard, and reads somewhere else.

let glb = 0;

function churn(): number {
    // enough traffic to hand a freed cell to somebody else
    let n = 0;
    for (let i = 0; i < 400; i++) {
        const s = "filler " + i;
        n = n + s.length;
    }

    return n;
}

function nestedLambdaSeesOuterLocal() {
    glb = 0;
    let a = 7;

    const outer = () => {
        const inner = () => { glb = glb + a; };
        inner();
    };

    outer();
    churn();
    assert(glb == 7, "nested lambda saw the outer local");
    assert(a == 7, "the outer local survived the inner box");
}

function threeLevelsDeep() {
    glb = 0;
    let a = 3;

    const one = () => {
        const two = () => {
            const three = () => { glb = glb + a; };
            three();
        };
        two();
    };

    one();
    churn();
    assert(glb == 3, "three levels of capture");
    assert(a == 3, "the outer local survived three boxes");
}

function innerCapturesTwoLevelsUp() {
    // the middle closure never names `a` itself - it only carries it through
    glb = 0;
    let a = 11;
    let b = 5;

    const outer = () => {
        glb = glb + b;
        const inner = () => { glb = glb + a; };
        inner();
    };

    outer();
    churn();
    assert(glb == 16, "the inner closure reached two frames up");
    assert(a == 11, "the skipped-over local survived");
}

function nestedLambdaMutatesOuterLocal() {
    let a = 1;

    const outer = () => {
        const inner = () => { a = a + 40; };
        inner();
        a = a + 1;
    };

    outer();
    churn();
    assert(a == 42, "the inner closure wrote through to the outer local");
}

function innerClosureOutlivesTheOuterCall() {
    // The inner closure is handed back rather than called on the spot, so its box is a real
    // one that nothing can fold away, and it is still holding the cell after the call that
    // built it has returned.
    glb = 0;
    let a = 2;

    const outer = () => {
        const inner = () => { glb = glb + a; };
        return inner;
    };

    const kept = outer();
    churn();
    kept();
    churn();
    kept();

    assert(glb == 4, "the escaped inner closure still reached the cell");
    assert(a == 2, "and the cell still holds its value");
}

function nestedLambdaOverAString() {
    // the cell holds something that owns heap memory of its own
    let s = "held";

    const outer = () => {
        const inner = () => { s = s + "!"; };
        return inner;
    };

    const kept = outer();
    churn();
    kept();
    churn();
    kept();
    churn();

    assert(s == "held!!", "the string in the captured cell survived");
}

function main() {
    nestedLambdaSeesOuterLocal();
    threeLevelsDeep();
    innerCapturesTwoLevelsUp();
    nestedLambdaMutatesOuterLocal();
    innerClosureOutlivesTheOuterCall();
    nestedLambdaOverAString();

    print("done.");
}
