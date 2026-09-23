// A try/catch written inside a catch clause whose own try also has a finally.
//
// Without the finally this already worked (00nested_catch.ts): a throw inside the nested try
// ends the outer catch ahead of itself, so the nested try runs outside the outer catch's funclet.
// With a finally, throws in the catch clause are left inside the funclet instead, because they
// invoke into the finally and the finally is what ends the catch. A throw inside the *nested*
// try, though, unwinds to the nested try's own pad, and that pad is emitted at the top level -
// so an invoke inside the catch funclet unwound to a pad outside it. That is invalid funclet
// nesting, and even a nested catch that matched crashed the program. Such a throw now ends the
// outer catch first, as it does without a finally.
//
// Also: an exception the nested catch does not take has to run the outer finally before it
// leaves the function. The nested try is in the outer try's catch clause, so its exceptions are
// not the outer try's to catch, but the outer finally still owes them a run.

let trail = 0;

function step(n: number) {
    trail = trail * 10 + n;
}

function nestedMatches() {
    try {
        throw 1;
    } catch (e: int) {
        step(1);
        try {
            throw 2;
        } catch (x: int) {
            step(9);
        }
        step(8);
    } finally {
        step(2);
    }
}

function nestedMismatchRunsOuterFinally() {
    try {
        throw 1;
    } catch (e: int) {
        step(1);
        try {
            throw "s";
        } catch (x: int) {
            step(9);
        }
        step(8);
    } finally {
        step(2);
    }
}

function bothFinallysRunInnermostFirst() {
    try {
        throw 1;
    } catch (e: int) {
        step(1);
        try {
            throw "s";
        } catch (x: int) {
            step(9);
        } finally {
            step(5);
        }
        step(8);
    } finally {
        step(2);
    }
}

function throwAfterNestedTry() {
    try {
        throw 1;
    } catch (e: int) {
        step(1);
        try {
            throw 2;
        } catch (x: int) {
            step(9);
        }
        throw "t";
    } finally {
        step(2);
    }
}

function run(f: () => void) {
    trail = 0;
    try {
        f();
    } catch (e: string) {
        step(3);
    }

    return trail;
}

function main() {
    assert(run(nestedMatches) == 1982, "a matching nested catch inside a catch with a finally");
    assert(run(nestedMismatchRunsOuterFinally) == 123, "a nested mismatch runs the outer finally, then reaches the caller");
    assert(run(bothFinallysRunInnermostFirst) == 1523, "both finally blocks run, innermost first");
    assert(run(throwAfterNestedTry) == 1923, "a throw after a nested try still runs the finally");

    print("done.");
}
