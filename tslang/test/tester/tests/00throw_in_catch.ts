// Throwing out of a catch clause has to end the active catch on the way. It is the same debt
// a `return`, `break` or `continue` leaving a catch already paid; a throw did not, and the
// resulting IR - a catchret emitted ahead of a call still carrying the funclet token - crashed
// the backend.
//
// JIT only, deliberately: an exception that escapes a catch clause is lost under AOT, and
// always was. A call inside a catch that throws (`catch (e) { thrower(); }`) loses it too,
// with no `throw` statement in the catch anywhere and nothing here able to affect it, so the
// gap is in the AOT exception tables rather than in what this file covers. The emitted IR is
// well-formed at -O0 and -O3; it is the runtime side that drops it.
//
// Known still-broken and deliberately not covered here: a *call* inside a catch clause
// followed by a throw out of it (`catch (e) { new Res(); throw 2; }`) crashes at run time.
// Separate bug, unrelated to ending the catch - that IR is well-formed too.

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
