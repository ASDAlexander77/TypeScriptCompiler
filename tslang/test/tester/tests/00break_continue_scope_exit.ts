// A `break` or `continue` owes every scope between itself and the loop it leaves - the
// disposals a `using` declared, and the references those scopes' locals took. It paid neither
// whenever it was written inside another block, which is where a `break` or `continue` is
// usually written: inside an `if`.
//
// The walk outwards stopped at the first scope that was not itself a loop, and `isLoop` reads
// true for every scope nested inside a loop, not just the loop's own body - so the very first
// step of the walk thought it had already arrived. Written directly in the loop body it
// happened to be right, which is why the shape below with no `if` around it always worked.
//
// Found by the ownership verifier (--verify-ownership) on its first run over the suite.
// 00owned_locals.ts already had the shape and asserted only counts, which a missed dispose
// does not change. See docs/reference-counting-evaluation.md section 9.18.

let disposed = 0;

class Res {
    [Symbol.dispose]() {
        disposed = disposed + 1;
    }
}

// `continue` from inside an `if`: the using scope is the loop body, one level out
function continueFromIf() {
    for (let i = 0; i < 3; i++) {
        using r = new Res();
        if (i == 1) {
            continue;
        }
    }
}

// `break` likewise, reaching the loop body twice before leaving
function breakFromIf() {
    for (let i = 0; i < 3; i++) {
        using r = new Res();
        if (i == 1) {
            break;
        }
    }
}

// two levels of nesting between the `continue` and the loop
function continueFromNestedBlock() {
    for (let i = 0; i < 3; i++) {
        using r = new Res();
        {
            if (i == 1) {
                continue;
            }
        }
    }
}

// the control: written directly in the loop body, which always worked
function continueDirect() {
    for (let i = 0; i < 3; i++) {
        using r = new Res();
        continue;
    }
}

// a `using` in the intermediate scope too - both owe a dispose on the way out
function bothScopes() {
    for (let i = 0; i < 2; i++) {
        using outer = new Res();
        if (i == 0) {
            using inner = new Res();
            continue;
        }
    }
}

// the labelled form still stops at the loop it names and not before it
function labelledContinue() {
    let rounds = 0;
    outer: while (rounds < 2) {
        rounds++;
        using a = new Res();
        let j = 2;
        while (j-- > 0) {
            using b = new Res();
            continue outer;
        }
    }

    return rounds;
}

function main() {
    disposed = 0;
    continueFromIf();
    assert(disposed == 3, "every iteration disposes, including the one that continues");

    disposed = 0;
    breakFromIf();
    assert(disposed == 2, "both iterations reached dispose, including the one that breaks");

    disposed = 0;
    continueFromNestedBlock();
    assert(disposed == 3, "two levels of nesting between the continue and the loop");

    disposed = 0;
    continueDirect();
    assert(disposed == 3, "the control: continue directly in the loop body");

    disposed = 0;
    bothScopes();
    assert(disposed == 3, "the intermediate scope's using disposes too");

    disposed = 0;
    assert(labelledContinue() == 2, "the labelled continue reaches its own loop");
    assert(disposed == 4, "and disposes both scopes it left, each round");

    print("done.");
}
