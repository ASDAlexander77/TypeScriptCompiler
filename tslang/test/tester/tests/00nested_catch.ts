// A `try`/`catch` written inside a `catch` clause. Two independent bugs made this crash - in
// every memory model, at every optimisation level, with nothing allocated in it - and the second
// one only became visible once the first was fixed.
//
// 1. The affine lowering finds a try's catch variable by walking the catches region, and the walk
//    descended into a nested try. It picked up the *inner* clause's catch, which set the RTTI
//    type filter on the outer try's landing pad from the wrong clause.
// 2. A catch clause can be ended twice over. The inner `throw` ends the enclosing catch ahead of
//    itself, and the outer try then emits its own end-of-catch marker as well; the surplus one
//    became the region's end instruction and survived into the emitted code, where an Itanium
//    `__cxa_end_catch` has no Win64 counterpart and fails to link.
//
// A try nested in a try *body* or in a `finally` always worked - only the catch clause was
// affected, which is why nothing caught this.
//
// These check which clause runs and in what order, and deliberately never read a catch
// variable's value. That is not squeamishness about the fix: reading one is broken on its own,
// with no nesting involved - `try { throw 2 } catch (v: int) { t = v }` in a module that throws
// only that one type reads 0 rather than 2, and reads correctly again once the module throws
// other types elsewhere. `00try_catch.ts` passes for that second reason. A separate bug, and
// nothing here should be made to depend on it.
//
// See docs/reference-counting-evaluation.md section 9.29.

type int = TypeOf<1>;

// the plain shape: a catch inside a catch
function catchInCatch() {
    let total = 0;
    try {
        throw 1;
    }
    catch (e: int) {
        try {
            throw 2;
        }
        catch (inner: int) {
            total = 11;
        }
    }

    return total;
}

// The outer try's landing pad must filter on its OWN clause's type. This is the direct test of
// bug 1: the two clauses take different types, so an outer pad carrying the inner clause's `int`
// filter would not catch the string at all and it would escape the function.
function outerFilterIsItsOwn() {
    let seen = 0;
    try {
        throw "outer";
    }
    catch (e: string) {
        seen = 1;
        try {
            throw 2;
        }
        catch (inner: int) {
            seen = seen + 10;
        }
    }

    return seen;
}

// three deep, so the fix is not a special case for one level
function threeDeep() {
    let total = 0;
    try {
        throw 1;
    }
    catch (a: int) {
        try {
            throw 2;
        }
        catch (b: int) {
            try {
                throw 3;
            }
            catch (c: int) {
                total = 100;
            }

            total = total + 10;
        }

        total = total + 1;
    }

    return total;
}

// the inner try does not throw at all, so its catch never runs
function innerCatchNotTaken() {
    let total = 0;
    try {
        throw 1;
    }
    catch (e: int) {
        try {
            total = 5;
        }
        catch (inner: int) {
            total = 99;
        }
    }

    return total;
}

// a nested try inside a catch, with a finally of its own
function nestedTryFinallyInCatch() {
    let total = 0;
    try {
        throw 1;
    }
    catch (e: int) {
        try {
            throw 2;
        }
        catch (inner: int) {
            total = 7;
        }
        finally {
            total = total + 1;
        }
    }

    return total;
}

// the whole thing inside a loop, so the regions are entered repeatedly
function catchInCatchInLoop() {
    let total = 0;
    for (let i = 0; i < 3; i++) {
        try {
            throw 1;
        }
        catch (e: int) {
            try {
                throw 2;
            }
            catch (inner: int) {
                total = total + 2;
            }
        }
    }

    return total;
}

// the inner clause throws on, past the outer clause, out of the function
function nestedCatchThrowsOn() {
    try {
        throw 1;
    }
    catch (e: int) {
        try {
            throw 2;
        }
        catch (inner: int) {
            throw 3;
        }
    }

    return 0;
}

// the shapes that always worked, kept alongside so a fix here cannot quietly break them
function tryInTryBody() {
    let total = 0;
    try {
        try {
            throw 2;
        }
        catch (inner: int) {
            total = 11;
        }
    }
    catch (e: int) {
        total = 1;
    }

    return total;
}

function tryInFinally() {
    let total = 0;
    try {
        total = 1;
    }
    finally {
        try {
            throw 2;
        }
        catch (inner: int) {
            total = total + 10;
        }
    }

    return total;
}

function main() {
    assert(catchInCatch() == 11, "a catch inside a catch runs");
    assert(outerFilterIsItsOwn() == 11, "the outer clause keeps its own type filter");
    assert(threeDeep() == 111, "three levels of catch nesting all run");
    assert(innerCatchNotTaken() == 5, "a nested try whose catch is not taken still runs its body");
    assert(nestedTryFinallyInCatch() == 8, "a nested try in a catch runs its own finally");
    assert(catchInCatchInLoop() == 6, "entering the nested regions repeatedly is fine");

    let rethrown = false;
    try {
        nestedCatchThrowsOn();
    }
    catch (e: int) {
        rethrown = true;
    }

    assert(rethrown, "a nested catch clause can throw past the outer one");

    assert(tryInTryBody() == 11, "a try in a try body still works");
    assert(tryInFinally() == 11, "a try in a finally still works");

    print("done.");
}
