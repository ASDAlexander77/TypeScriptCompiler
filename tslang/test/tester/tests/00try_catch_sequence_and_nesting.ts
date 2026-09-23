// try/catch shapes that each failed on Windows, all of them silently or with a crash at run time,
// none of them at compile time. Each function exercises one of them.
//
// - Two try/catch statements in a row, the second one's body starting with a call. The
//   canonicalizer merges the first catch block into what follows it, so the first instruction
//   after the first catch's end marker was the second try's `invoke`. Win32ExceptionPass took
//   that invoke for the catch's own end-of-catch invoke and gave the catch no catchret, so the
//   second try ran inside the first catch's funclet and the process crashed.
//
// - Two different classes thrown from one module. Every class shared one ThrowInfo and one
//   CatchableTypeArray name (`_TI2PEAV` / `_CTA2PEAV`), both linkonce_odr, so the first class
//   thrown fixed the descriptors for every other: `throw new B()` went out as an `A`.
//
// - A typed catch that does not match, nested in another try in the same function. On Windows
//   the catchswitch learns where to unwind next only from an invoke inside its catch body; with
//   no call there the enclosing try's landing pad had no predecessor, was deleted, and the
//   exception left the function.
//
// - The same mismatch with a `finally` on the inner try: that finally was skipped.
//
// The last two failed on Linux as well, for an Itanium reason: the inner landingpad listed only
// its own typed clause, so the personality never entered it for another type, and with nothing
// in the frame unwinding to the finally or the outer try the program called std::terminate.
//
// The value a catch binds is read only for typed class catches, which 00catch_value.ts covers.

class A {
    a = 1;
}

class B {
    b = 2;
}

class Base {
    k = 3;
}

class Derived extends Base {
    d = 4;
}

function one() {
    return 1;
}

function sequentialTryWithCall() {
    let steps = 0;
    try {
        throw 1;
    } catch (e: int) {
        steps += 1;
    }

    try {
        one();
        throw 2;
    } catch (e2: int) {
        steps += 10;
    }

    try {
        let a = new A();
        throw 3;
    } catch (e3: int) {
        steps += 100;
    }

    return steps;
}

function twoClassesThrown() {
    let sum = 0;
    try {
        throw new A();
    } catch (e: A) {
        sum += e.a;
    }

    try {
        throw new B();
    } catch (e2: B) {
        sum += e2.b * 10;
    }

    try {
        throw new Derived();
    } catch (e3: Base) {
        sum += e3.k * 100;
    }

    return sum;
}

function nestedMismatchPrimitive() {
    let path = 0;
    try {
        try {
            throw 1;
        } catch (e: string) {
            path = 1;
        }
    } catch (e2: int) {
        path += 10;
    }

    return path;
}

function nestedMismatchClass() {
    let path = 0;
    try {
        try {
            throw new B();
        } catch (e: A) {
            path = 1;
        }
    } catch (e2: B) {
        path += 10 * e2.b;
    }

    return path;
}

function nestedMatchStaysInner() {
    let path = 0;
    try {
        try {
            throw 1;
        } catch (e: int) {
            path += 1;
        }
    } catch (e2: int) {
        path += 10;
    }

    return path;
}

function threeDeepMismatch() {
    let path = 0;
    try {
        try {
            try {
                throw 1;
            } catch (e: string) {
                path += 1;
            }
        } catch (e2: B) {
            path += 10;
        }
    } catch (e3: int) {
        path += 100;
    }

    return path;
}

function mismatchRunsOwnFinally() {
    let path = 0;
    try {
        try {
            throw 1;
        } catch (e: string) {
            path += 1;
        } finally {
            path += 10;
        }
    } catch (e2: int) {
        path += 100;
    }

    return path;
}

let trail = 0;

function mismatchInCalleeRunsFinally() {
    try {
        throw 1;
    } catch (e: string) {
        trail = trail * 10 + 1;
    } finally {
        trail = trail * 10 + 2;
    }
}

function callerCatchesAfterFinally() {
    trail = 0;
    try {
        mismatchInCalleeRunsFinally();
    } catch (e: int) {
        trail = trail * 10 + 3;
    }

    return trail;
}

function main() {
    assert(sequentialTryWithCall() == 111, "every catch in a sequence of try/catch statements runs and returns");
    assert(twoClassesThrown() == 321, "each thrown class reaches its own typed catch");
    assert(nestedMismatchPrimitive() == 10, "a nested catch that does not match leaves the exception to the outer try");
    assert(nestedMismatchClass() == 20, "a nested class catch that does not match leaves the exception to the outer try");
    assert(nestedMatchStaysInner() == 1, "a nested catch that matches keeps the exception");
    assert(threeDeepMismatch() == 100, "an exception passes two non-matching catches to reach a third");
    assert(mismatchRunsOwnFinally() == 110, "a catch that does not match still runs its try's finally");
    assert(callerCatchesAfterFinally() == 23, "the finally runs before the caller's catch");

    print("done.");
}
