// regression tests for break/continue interacting with try/finally:
// - break/continue directly inside `finally` must keep correct wiring in all copies of
//   the cloned finally region (TryOpLowering clones it for the exception/return paths);
// - break/continue inside the try *body* or a catch clause that jumps out of the try must
//   run the enclosing `finally` on the way out, the same way `return` does -- routed
//   through a dedicated finally copy per escape target;
// - nested try/finally chains must run every finally on the way out, innermost first.

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

function test_break_in_try_body() {
    let iterations = 0;
    let finallyRuns = 0;

    for (let i = 0; i < 5; i++) {
        iterations++;
        try {
            if (i == 2) {
                break;
            }
        } finally {
            finallyRuns++;
        }
    }

    assert(iterations == 3, "break-in-body: wrong iteration count");
    assert(finallyRuns == 3, "break-in-body: finally must run on the break iteration too");
    return 1;
}

function test_continue_in_try_body() {
    let sum = 0;
    let finallyRuns = 0;

    for (let i = 0; i < 5; i++) {
        try {
            if (i == 2) {
                continue;
            }
            sum += i;
        } finally {
            finallyRuns++;
        }
    }

    assert(finallyRuns == 5, "continue-in-body: finally must run on the continue iteration too");

    // sum skips i == 2: 0 + 1 + 3 + 4 = 8
    return sum;
}

function test_nested_finally_chain() {
    let f1 = 0;
    let f2 = 0;
    let iterations = 0;

    for (let i = 0; i < 5; i++) {
        iterations++;
        try {
            try {
                if (i == 1) {
                    break;
                }
            } finally {
                f1++;
            }
        } finally {
            f2++;
        }
    }

    assert(iterations == 2, "nested-chain: wrong iteration count");
    assert(f1 == 2, "nested-chain: inner finally skipped");
    assert(f2 == 2, "nested-chain: outer finally skipped");
    return 1;
}

function test_break_in_catch() {
    let caught = 0;
    let finallyRuns = 0;
    let iterations = 0;

    for (let i = 0; i < 5; i++) {
        iterations++;
        try {
            if (i == 2) {
                throw "boom";
            }
        } catch (e: string) {
            caught++;
            break;
        } finally {
            finallyRuns++;
        }
    }

    assert(iterations == 3, "break-in-catch: wrong iteration count");
    assert(caught == 1, "break-in-catch: catch did not run");
    assert(finallyRuns == 3, "break-in-catch: finally must run when catch breaks out");
    return 1;
}

function test_continue_and_break_two_targets() {
    // the same try has both a break and a continue escaping -- two distinct targets,
    // each must get its own finally routing
    let sum = 0;
    let finallyRuns = 0;

    for (let i = 0; i < 10; i++) {
        try {
            if (i % 2 == 0) {
                continue;
            }
            if (i == 7) {
                break;
            }
            sum += i;
        } finally {
            finallyRuns++;
        }
    }

    // odd i below 7 summed: 1 + 3 + 5 = 9; loop left at i == 7
    assert(sum == 9, "two-targets: wrong sum");
    assert(finallyRuns == 8, "two-targets: finally runs for i = 0..7");
    return 1;
}

function test_inner_loop_break_untouched() {
    // break targeting a loop fully inside the try must NOT be routed through the
    // finally -- it doesn't escape the try
    let finallyRuns = 0;
    let innerRuns = 0;

    try {
        for (let i = 0; i < 10; i++) {
            if (i == 3) {
                break;
            }
            innerRuns++;
        }
    } finally {
        finallyRuns++;
    }

    assert(innerRuns == 3, "inner-loop: wrong inner count");
    assert(finallyRuns == 1, "inner-loop: finally must run exactly once");
    return 1;
}

function main() {
    assert(test_break_in_finally() == 33, "failed. 1");
    assert(test_continue_in_finally() == 8, "failed. 2");
    assert(test_nested_try_in_finally() == 3, "failed. 3");
    assert(test_break_in_try_body() == 1, "failed. 4");
    assert(test_continue_in_try_body() == 8, "failed. 5");
    assert(test_nested_finally_chain() == 1, "failed. 6");
    assert(test_break_in_catch() == 1, "failed. 7");
    assert(test_continue_and_break_two_targets() == 1, "failed. 8");
    assert(test_inner_loop_break_untouched() == 1, "failed. 9");

    print("done.");
}
