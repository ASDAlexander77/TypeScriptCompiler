// A reference-counted program built WITH debug info. Registered with `--di`, which no other
// test in this suite passes alongside `-mm=rc` - and that gap is why the bug below lasted from
// section 9.31 to section 9.69 with a one-line note and no test.
//
// It was not a wrong answer, it was no answer: `--di --opt_level=0 -mm=rc` failed to emit LLVM
// IR at all, for every reference-counted program, with "DISubprogram attached to more than one
// function". Every op in a generated ownership routine (`tsrel_`, `tsret_`, `__tslang_inc_ref`,
// `__tslang_dec_ref`, `__tslang_free_block`) is built with the location of whatever op triggered
// its generation, because a routine synthesised from a type has no source of its own - and under
// `--di` that location carries the enclosing user function's DISubprogram. The routines are
// rc-only, so `gc` and `none` never saw it.
//
// The shapes below are chosen to make the compiler generate those routines: a string local, an
// owning field, an array, a closure over a captured variable, and a class instance. Each needs a
// release routine, and between them they reach the retain routines and all three `__tslang_*`
// helpers. What is asserted is ordinary behaviour - the point of the test is that it compiles.

class Holder {
    name: string;
    constructor(name: string) { this.name = name; }
}

function ownsAString() {
    let s = "value";
    return s.length;
}

function ownsAField() {
    let h = new Holder("field");
    return h.name.length;
}

function ownsAnArray() {
    let a = ["one", "two"];
    a.push("three");
    return a.length;
}

function ownsACapture() {
    let seen = 0;
    let bump = () => { seen = seen + 1; };
    bump();
    bump();
    return seen;
}

function main() {
    assert(ownsAString() == 5, "a string local");
    assert(ownsAField() == 5, "a field that owns a string");
    assert(ownsAnArray() == 3, "an array that owns its elements");
    assert(ownsACapture() == 2, "a closure over a captured variable");

    print("done.");
}
