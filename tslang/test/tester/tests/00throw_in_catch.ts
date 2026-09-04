// Throwing out of a catch clause has to end the active catch on the way. It is the same debt
// a `return`, `break` or `continue` leaving a catch already paid; a throw did not, and the
// resulting IR - a catchret emitted ahead of a call still carrying the funclet token - crashed
// the backend.
//
// This file was JIT-only when it was written, for three reasons that have all since gone
// away, and none of which were about ending the catch:
//
//  - An exception escaping a catch clause was said to be lost under AOT. It was the
//    CatchableType::sizeOrOffset miscompile, fixed in the commit after this file landed; see
//    docs/reference-counting-evaluation.md section 9.15.
//  - So was `catch (e) { new Res(); throw 2; }` crashing at run time - a call in a catch
//    followed by a throw out of it. Same fix, same reason: a frame slot overwritten by a
//    caught `int` copied as 8 bytes.
//  - `catch (e) { thrower(); }`, a call in a catch that throws with no `throw` statement
//    anywhere, was blamed on the AOT exception tables. It was neither AOT-specific nor
//    exception-table-related: the MLIR inliner was erasing the throw. 00throw_inlined.ts
//    covers it, and section 9.16 has the detail.

function throwsALiteralFromCatch() {
    try {
        throw 1;
    }
    catch (e: TypeOf<1>) {
        throw 2;
    }
}

// the rethrow idiom: `catch (e) { throw e; }`
function rethrows() {
    try {
        throw 7;
    }
    catch (e: TypeOf<1>) {
        throw e;
    }
}

// the catch that throws is itself nested inside another try, so the new exception must reach
// the outer handler and not be re-caught by the one it was thrown from
function nestedThrowFromCatch() {
    let reached = 0;
    try {
        try {
            throw 1;
        }
        catch (e: TypeOf<1>) {
            reached = reached + 1;
            throw 2;
        }
    }
    catch (e: TypeOf<1>) {
        reached = reached + 10;
    }

    return reached;
}

// a catch that does not throw still ends normally
function plainCatch() {
    let ran = 0;
    try {
        throw 1;
    }
    catch (e: TypeOf<1>) {
        ran = 1;
    }

    return ran;
}

function caught(f: () => void) {
    try {
        f();
    }
    catch (e: TypeOf<1>) {
        return true;
    }

    return false;
}

function main() {
    assert(caught(() => throwsALiteralFromCatch()), "a throw from a catch must reach the caller");
    assert(caught(() => rethrows()), "a rethrow from a catch must reach the caller");
    assert(nestedThrowFromCatch() == 11, "the outer handler must take it, and the inner one must not re-catch");
    assert(plainCatch() == 1, "a catch that does not throw still runs and ends normally");

    print("done.");
}
