// Locals declared in catch and finally clauses own what they hold: released where the clause ends,
// on a return or break out of it, and kept alive (not freed early) under every model. One that an
// exception leaves the clause with leaks: the clause has no cleanup of its own.
class R { constructor(public v: number, public s: string) {} }
@noinline function mk(i: number): R { return new R(i, "r" + i); }
function throwOutOfCatch(i: number): number {
    try {
        try { throw 1; } catch (e) { const k = mk(i); if (k.v >= 0) throw 2; return k.v; }
    } catch (e2) { return 7; }
}
function returnFromFinally(i: number): number {
    try { const a = mk(i); } finally { const k = mk(i + 1); return k.v; }
}
function breakFromFinally(n: number): number {
    let s: number = 0;
    for (let i = 0; i < n; i++) {
        try { s += i; } finally { const k = mk(i); s += k.v; if (i == 3) break; }
    }
    return s;
}
function nestedTryInCatch(i: number): number {
    try { throw 1; } catch (e) {
        const k = mk(i);
        try { const j = mk(i + 1); throw 3; } catch (e3) { const m = mk(i + 2); return k.v + m.v; }
    }
}
function rethrow(i: number): number {
    try {
        try { throw 1; } catch (e) { const k = mk(i); throw e; }
    } catch (e4) { return 9; }
}
function finallyAfterThrow(i: number): number {
    let r: number = 0;
    try {
        try { throw 5; } finally { const k = mk(i); r = k.v; }
    } catch (e5) { r += 100; }
    return r;
}
function main() {
    for (let i = 0; i < 300; i++) {
        assert(throwOutOfCatch(i) == 7, "throwOutOfCatch");
        assert(returnFromFinally(i) == i + 1, "returnFromFinally");
        assert(breakFromFinally(10) == 12, "breakFromFinally");
        assert(nestedTryInCatch(i) == 2 * i + 2, "nestedTryInCatch");
        assert(rethrow(i) == 9, "rethrow");
        assert(finallyAfterThrow(i) == i + 100, "finallyAfterThrow");
    }
    print("done.");
}
