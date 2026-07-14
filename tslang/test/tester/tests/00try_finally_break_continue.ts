// regression test for TryOpLowering's finally-region cloning: break/continue and a
// nested try inside `finally` must keep correct wiring in all 3 copies of the region
// (the two cloned exception-path copies, and the inlined normal-exit copy).
//
// Note: break/continue inside a try *body* (rather than directly in `finally`) is a
// separate, pre-existing bug tracked in docs/LowerToAffineLoops-LowerToLLVM-review.md --
// intervening finally blocks are not run before the jump, so those cases are not covered
// here.

function test_break_in_finally() {
    let iterations = 0;
    let finallyRuns = 0;

    for (let i = 0; i < 5; i++) {
        iterations++;
        try {
            print("try ", i);
        } finally {
            finallyRuns++;
            if (i == 2) {
                break;
            }
        }
    }

    assert(iterations == 3, "break-in-finally: wrong iteration count");
    assert(finallyRuns == 3, "break-in-finally: wrong finally-run count");

    return iterations * 10 + finallyRuns;
}

function test_continue_in_finally() {
    let sum = 0;
    let finallyRuns = 0;

    for (let i = 0; i < 5; i++) {
        try {
            print("try ", i);
        } finally {
            finallyRuns++;
            if (i == 2) {
                continue;
            }
            sum += i;
        }
    }

    assert(finallyRuns == 5, "continue-in-finally: wrong finally-run count");

    // sum should skip i == 2 (continue in finally happens before sum += i runs for that
    // iteration): 0 + 1 + 3 + 4 = 8
    return sum;
}

function test_nested_try_in_finally() {
    let caught = 0;
    let finallyRuns = 0;

    for (let i = 0; i < 3; i++) {
        try {
            print("outer try ", i);
        } finally {
            finallyRuns++;
            try {
                throw "inner";
            } catch (e: string) {
                caught++;
            }
        }
    }

    assert(finallyRuns == 3, "nested-try-in-finally: wrong finally-run count");
    assert(caught == 3, "nested-try-in-finally: inner catch did not run every time");

    return caught;
}

function main() {
    assert(test_break_in_finally() == 33, "failed. 1");
    assert(test_continue_in_finally() == 8, "failed. 2");
    assert(test_nested_try_in_finally() == 3, "failed. 3");

    print("done.");
}
