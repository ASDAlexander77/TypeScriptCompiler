// Boxing a value into `any` allocates a box and copies the value into it, and the box owns what
// it holds: when it dies it releases the payload through the type tag beside it. Nothing was
// taking that reference, so a boxed value that arrived carrying one - a call's result, or a
// closure and the box of captured variables it was built over - had that reference given back at
// the end of the block that boxed it, leaving the `any` pointing at freed memory.
//
// Every case below builds the `any` in one function and reads it in another, with `churn()`
// between, because within one block the release that causes it lands after the read and the case
// passes with the bug present. Reading through a freed-then-reused block is what fails.
//
// See docs/reference-counting-evaluation.md section 9.36.

type reader = () => number;

class Vec {
    x: number;

    constructor(x: number) {
        this.x = x;
    }
}

// Allocate over whatever has just been freed, so a use-after-free reads something else.
function churn() {
    for (let i = 0; i < 64; i++) {
        let filler = new Vec(999);
    }
}

// The general shape, and the one that says this is not about closures: a call's result carries a
// reference for its receiver, and the box is that receiver.
function makeName(): string {
    return "na" + "me";
}

function boxCallResult(): any[] {
    let out: any[] = [];
    out.push(makeName());

    return out;
}

function callResultBoxedAsAny() {
    let boxed = boxCallResult();
    churn();

    return (<string>boxed[0]).length;
}

// The same for a closure, where what is freed is the capture box rather than the value itself,
// so the wrong answer comes back through the captured variable.
function boxClosure(k: number): any[] {
    const kk = k;
    let out: any[] = [];
    out.push(qux);

    return out;

    function qux() {
        return kk;
    }
}

function closureBoxedAsAny() {
    let boxed = boxClosure(22);
    churn();

    return (<reader>boxed[0])();
}

// Not through an array: a single `any` field outlives its block just as an element does, and
// reaches the same boxing cast.
class AnyHolder {
    item: any;

    constructor(item: any) {
        this.item = item;
    }
}

function boxIntoField(k: number): AnyHolder {
    const kk = k;

    return new AnyHolder(qux);

    function qux() {
        return kk;
    }
}

function closureBoxedIntoField() {
    let holder = boxIntoField(33);
    churn();

    return (<reader>holder.item)();
}

// A boxed value the frame also still holds: the box takes a reference of its own, so neither
// owner freeing is the other's problem, and the string outlives the shorter of the two.
function boxSharedString(): any[] {
    let s = makeName();
    let out: any[] = [];
    out.push(s);

    return out;
}

function stringBoxedAndStillHeld() {
    let boxed = boxSharedString();
    churn();

    return (<string>boxed[0]).length;
}

// Two boxes over the same constant are two blocks. Boxing is an allocation, so two structurally
// identical boxing casts are not one value: CSE merged them, and the second assignment then stored
// a pointer to the block the first had already released. The read below is what that costs - the
// box is freed once at the assignment that replaced it and again at the end of the block, and in
// between something else is allocated over it.
//
// Only a constant reaches this shape: two reads of a runtime value are two `ts.Load`s, so the
// casts over them are not identical and CSE leaves them alone. A constant is CSE'd into one op
// first, which is what makes the casts over it identical.
//
// See docs/reference-counting-evaluation.md section 9.48.
function refillStrings() {
    for (let i = 0; i < 64; i++) {
        let filler: any = "xy";
    }
}

function refillNumbers() {
    for (let i = 0; i < 64; i++) {
        let filler: any = 9.5;
    }
}

function sameStringBoxedTwice(): number {
    let a: any = "abcd";
    a = 5;
    a = "abcd";
    refillStrings();

    return (<string>a).length;
}

function sameNumberBoxedTwice(): number {
    let a: any = 4.5;
    a = "x";
    a = 4.5;
    refillNumbers();

    return <number>a;
}

function main() {
    assert(sameStringBoxedTwice() == 4, "boxing the same string twice makes two boxes");
    assert(sameNumberBoxedTwice() == 4.5, "boxing the same number twice makes two boxes");
    assert(callResultBoxedAsAny() == 4, "an `any` owns what a call handed it");
    assert(closureBoxedAsAny() == 22, "an `any` owns the closure boxed into it");
    assert(closureBoxedIntoField() == 33, "an `any` field owns what was boxed into it");
    assert(stringBoxedAndStillHeld() == 4, "boxing takes a reference of its own");

    print("done.");
}
