// A yield in a finally block of a generator, as in JavaScript: the finally runs once on each way out of the try -
// its end, a throw, a `return` (from the try or the catch clause), a `break` or `continue` out of it, return() and
// throw() - and pauses the generator at its yields; then the way out is taken: the value returned, the exception
// thrown on, the jump made. return(v) while the generator is paused in the try gives what the finally yields, and v
// when it is done. The expected results are node's for the same JavaScript. Before, the finally was copied for each
// way out and a resume landed in one copy only: its yield was skipped, and the rest of it ran from the wrong place
// (then an error). Also: a `let` of a nested block hiding one around it in a generator shared its variable.
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    return: (value: T) => IterateResult<T>;
    throw: (e: any) => IterateResult<T>;
}

let log = "";

function* g1(): Generator<number> { try { yield 1; } finally { log += "f1,"; yield 2; log += "f1b,"; } yield 3; }
function* g2(): Generator<number> { try { yield 1; return 5; } finally { yield 2; } }
function* g3(): Generator<number> { try { yield 1; throw "e3"; } finally { yield 2; } }
function* g4(): Generator<number> { try { yield 1; throw "e4"; } catch (e) { log += "c:" + e + ","; return 6; } finally { yield 2; } }
function* g5(): Generator<number> { for (let i = 0; i < 3; i++) { try { if (i == 1) continue; if (i == 2) break; yield i; } finally { yield 10 + i; } } yield 99; }
function* g6(): Generator<number> { try { try { yield 1; return 7; } finally { yield 2; } } finally { yield 3; } }
function* g7(): Generator<number> { let x = 1; try { let x = 2; yield x; return x; } finally { yield x; } }
function* g8(): Generator<number> { try { yield 1; } finally { yield 2; throw "e8"; } }

function show(r: IterateResult<number>) {
    return r.done ? (r.value === undefined ? "D" : "D" + r.value) : "" + r.value;
}

function run(it: Generator<number>) {
    let s = "";
    try {
        for (;;) {
            const r = it.next();
            s += show(r) + " ";
            if (r.done) break;
        }
    } catch (e) {
        s += "throw:" + e;
    }
    return s;
}

function main() {
    assert(run(g1()) == "1 2 3 D0 " && log == "f1,f1b,", "end of the try");
    assert(run(g2()) == "1 2 D5 ", "return through it");
    assert(run(g3()) == "1 2 throw:e3", "throw through it");
    assert(run(g4()) == "1 2 D6 " && log == "f1,f1b,c:e4,", "return from the catch clause");
    assert(run(g5()) == "0 10 11 12 99 D0 ", "continue and break through it");
    assert(run(g6()) == "1 2 3 D7 ", "nested");
    assert(run(g7()) == "2 1 D2 ", "the finally reads the outer x");
    assert(run(g8()) == "1 2 throw:e8", "a throw after the finally's yield");
    {
        const it = g1(); it.next();
        assert(show(it.return(9)) == "2" && show(it.next()) == "D9" && show(it.next()) == "D0", "return() in the try");
    }
    {
        const it = g1(); it.next(); it.next();
        assert(show(it.return(9)) == "D9" && show(it.next()) == "D0", "return() in the finally");
    }
    {
        const it = g1(); it.next();
        let a = ""; try { a = show(it.throw("t")); } catch (e) { a = "throw:" + e; }
        let b = ""; try { b = show(it.next()); } catch (e) { b = "throw:" + e; }
        assert(a == "2" && b == "throw:t", "throw() in the try");
    }
    print("done.");
}
