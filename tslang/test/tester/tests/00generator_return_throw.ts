// return(value) and throw(e) of a generator typed Generator<T, ...>, as JavaScript's: one paused at a yield
// resumes there finishing (its finally blocks run) or throwing e (its own try/catch may catch it, and it goes on
// to its next yield); one not started yet, or finished, does not run. Fixed on the way: a throw or a call after a
// resumed yield inside a `try` went past it (its catch did not catch, its finally did not run), a `let` in a `try`
// read after a yield and a throw after a yield in a loop in a `try` did not compile, and a catch variable of a
// generator crashed. (-mm=own: return() only for a T that owns nothing - a number, a boolean.)
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    return: (value: T) => IterateResult<T>;
    throw: (e: any) => IterateResult<T>;
}

let cleaned = 0;
let lastCaught = "";

function* counter(): Generator<number> {
    try {
        let i = 0;
        while (true) yield i++;
    } finally {
        cleaned++;
    }
}

function* catcher(): Generator<number> {
    let caught = 0;
    while (true) {
        try {
            yield caught;
        } catch (e) {
            caught++;
            lastCaught = "caught " + e;
        }
    }
}

// no return()/throw() involved: the try fixes alone
function* afterYield() {
    try {
        let n = 1;
        yield n;
        if (n == 1) throw "after";
    } catch (e) {
        lastCaught = "caught " + e;
    }

    yield 2;
}

function main() {
    // return() of a paused generator: finishes, its finally runs
    const c = counter();
    assert(c.next().value == 0 && c.next().value == 1);
    const r = c.return(42);
    assert(r.value == 42 && r.done && cleaned == 1, "return() finishes, finally runs");
    assert(c.next().done, "finished after return()");

    // return() of one not started: does not run
    const n = counter();
    const r2 = n.return(7);
    assert(r2.value == 7 && r2.done && cleaned == 1, "return() before the start");
    assert(n.next().done);

    // throw() caught by the body: it goes on to its next yield
    const k = catcher();
    assert(k.next().value == 0);
    assert(k.throw("boom").value == 1, "throw() caught");
    assert(lastCaught == "caught boom");
    assert(k.next().value == 1);

    // throw() not caught: it comes out, the finally runs, the generator is finished
    const u = counter();
    u.next();
    let out = "";
    try {
        u.throw("out");
    } catch (e) {
        out = "thrown " + e;
    }

    assert(out == "thrown out" && cleaned == 2, "throw() not caught");
    assert(u.next().done);

    // throw() of one not started: thrown at once
    let early = "";
    try {
        counter().throw("early");
    } catch (e) {
        early = "thrown " + e;
    }

    assert(early == "thrown early" && cleaned == 2, "throw() before the start");

    let total = 0;
    for (const v of afterYield()) total += v;
    assert(total == 3 && lastCaught == "caught after", "a throw after a resumed yield is caught");

    print("done.");
}
