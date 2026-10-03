// A branch written without braces is a scope of its own (#458): what it declares and must give back
// at the end - the iterator of a `for ... of` over a generator, a `using` - is released at the end
// of the branch. It was handed to the block around the `if`, whose end is outside the branch, and
// the IR failed to verify: "operand #0 does not dominate this use".
function* gen() {
    yield 1;
    yield 2;
}

class Res {
    [Symbol.dispose]() {
        print("disposed");
    }
}

function main() {
    let sum = 0;
    const it = gen();
    if (false) it.next(); else for (const c of it) sum += c;
    assert(sum == 3, "an unbraced else that iterates a generator");

    for (let i = 0; i < 2; i++) {
        if (i == 0) sum += 10; else for (const c of gen()) sum += c;
    }

    assert(sum == 16, "the same inside a loop");

    if (sum > 0) for (const c of gen()) sum += c;
    assert(sum == 19, "an unbraced then");

    let total = 0;
    for (let i = 0; i < 3; i++) if (i != 1) for (const c of gen()) total += c;
    assert(total == 6, "an unbraced if as a loop body");

    print("done.");
}
