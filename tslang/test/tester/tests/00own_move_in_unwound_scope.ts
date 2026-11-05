// -mm=own: a local moved to another inside a block whose scope is also released on the unwind leg -
// a `try` body, or on Itanium (Linux) a function body that holds a typed `catch` - was "moved here on
// some paths only": the unwind leg's release is reached from before the move and after it. The
// moved-from slot is emptied at the move, so that release frees the value only if the move has not
// run. Each case runs many times so a double free or a use of a freed block shows.

class E {
    constructor(public m: string) {}
}

function make(n: int): int[] {
    return [n, n + 1];
}

function check(a: int[], fail: boolean): int {
    if (fail) {
        throw new E("check");
    }

    return a.length;
}

async function twice(n: int): Promise<int> {
    return n * 2;
}

async function pair(n: int): Promise<int[]> {
    return [n, n * 2];
}

// Promise.all's shape: the values, already settled
function all(values: int[]): int[] {
    const settled: int[] = [];
    for (const value of values) {
        settled.push(value);
    }

    return settled;
}

// the move in a `try` body; the throw before it, or after it, unwinds through the body's scope
function inTry(failBefore: boolean, failAfter: boolean): int {
    let n = 0;
    try {
        const a = make(1);
        check(a, failBefore);
        const b = a;
        n = check(b, failAfter);
    } catch (e: E) {
        n = -1;
    }

    return n;
}

// the move and a typed catch in one function body: on Itanium the body is wrapped for its cleanup
function besideCatch(fail: boolean): int {
    const first = make(1);
    const a = make(2);
    const b = a;
    let caught = false;
    try {
        const none: int[] = [];
        check(none, fail);
    } catch (e: E) {
        caught = true;
    }

    return first.length + b.length + (caught ? 10 : 0);
}

// an awaited value moved out of the await's slot, beside a typed catch
function awaitBesideCatch(fail: boolean): int {
    const first = make(1);
    const awaited = await pair(3);
    let caught = false;
    try {
        check(first, fail);
    } catch (e: E) {
        caught = true;
    }

    return awaited[0] + awaited[1] + (caught ? 100 : 0);
}

// the shape from DefaultLib's promise test: an array literal of async calls, awaited
function awaitAllBesideCatch(fail: boolean): int {
    const both = [twice(2), twice(5)];
    const awaited = await all([twice(3), twice(4)]);
    let caught = false;
    try {
        const none: int[] = [];
        check(none, fail);
    } catch (e: E) {
        caught = true;
    }

    return both[0] + awaited[0] + awaited[1] + (caught ? 100 : 0);
}

// the unwind leaves the function: its own scope's release on the way out, after the move
function moveThenThrow(): int {
    const a = make(4);
    const b = a;
    let caught = false;
    try {
        check(b, false);
    } catch (e: E) {
        caught = true;
    }

    check(b, true);
    return caught ? 1 : 0;
}

function main() {
    for (let i = 0; i < 1000; i++) {
        assert(inTry(false, false) == 2, "in try, no throw");
        assert(inTry(true, false) == -1, "in try, thrown before the move");
        assert(inTry(false, true) == -1, "in try, thrown after the move");

        assert(besideCatch(false) == 4, "beside a catch");
        assert(besideCatch(true) == 14, "beside a catch, caught");

        assert(awaitBesideCatch(false) == 9, "an await beside a catch");
        assert(awaitBesideCatch(true) == 109, "an await beside a catch, caught");

        assert(awaitAllBesideCatch(false) == 18, "an awaited array of async calls beside a catch");
        assert(awaitAllBesideCatch(true) == 118, "an awaited array of async calls beside a catch, caught");

        let caught = "";
        try {
            moveThenThrow();
        } catch (e: E) {
            caught = e.m;
        }

        assert(caught == "check", "moved, then thrown out of the function");
    }

    print("done.");
}
