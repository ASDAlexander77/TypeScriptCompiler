// An optional number compares as the wider of its own type and the other operand's - not as the
// other operand's alone, which made 0.5 an s32 against an integer literal - and only when it holds a
// value: an empty one is not 0, it is less than, greater than and equal to nothing
function holdsFraction() {
    let n: number | undefined = 0.5;
    assert(!(n === 0), "0.5 === 0");
    assert(n === 0.5, "0.5 === 0.5");
    assert(n !== 0, "0.5 !== 0");
    assert(n < 1, "0.5 < 1");
    assert(n > 0, "0.5 > 0");
    assert(!(0 === n), "0 === 0.5");
    assert(0.5 == n, "0.5 == 0.5");

    let zero: number = 0;
    let half: number = 0.5;
    assert(!(n === zero), "0.5 === zero");
    assert(n === half, "0.5 === half");
    assert(n > zero, "0.5 > zero");
}

function empty() {
    let n: number | undefined = undefined;
    assert(!(n === 0), "undefined === 0");
    assert(!(n == 0), "undefined == 0");
    assert(n !== 0, "undefined !== 0");
    assert(!(n < 1), "undefined < 1");
    assert(!(n >= 0), "undefined >= 0");
    assert(!(0 === n), "0 === undefined");

    let zero: number = 0;
    assert(!(n === zero), "undefined === zero");
    assert(!(n <= zero), "undefined <= zero");
}

function integers() {
    let i: s32 | undefined = 3;
    assert(i === 3, "s32 3 === 3");
    assert(!(i === 0), "s32 3 === 0");
    assert(i > 2, "s32 3 > 2");
    assert(i < 3.5, "s32 3 < 3.5");

    i = undefined;
    assert(!(i === 0), "s32 undefined === 0");
    assert(!(i < 1), "s32 undefined < 1");

    let f: number | undefined = 3;
    assert(f === 3, "3.0 === 3");
    assert(f > 2.5, "3.0 > 2.5");
}

function main() {
    holdsFraction();
    empty();
    integers();
    print("done.");
}
