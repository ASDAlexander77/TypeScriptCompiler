// `b && f || g` is `F | G` (`b && f` is `false | F`, as in TypeScript, and `||` drops the `false`), and
// `c ? f : g` / `o.f ?? g` of two functions is the one taking the most parameters: each can be called.
function none(): number { return 0; }
function one(a?: number): number { return a === undefined ? -1 : a; }
function zero(x: number): number { return 0; }
function unit(x: number): number { return 1; }

function pick(b: boolean) {
    const choice = b && zero || unit;
    return choice(0);
}

function cond(b: boolean) {
    const h = b ? one : none;
    return h(3);
}

function nullish(o: { f?: (a?: number) => number }) {
    const g = o.f ?? none;
    return g(5);
}

function main() {
    assert(pick(true) == 0);
    assert(pick(false) == 1);

    assert(cond(true) == 3);
    assert(cond(false) == 0);

    assert(nullish({}) == 0);
    assert(nullish({ f: one }) == 5);

    // `&&` of a boolean still gives `false` where the left side is false
    const n: number | undefined = 3;
    assert((n !== undefined && n * n || 0) == 9);
    const t = true;
    const f = !t;
    assert((f && 5) === false);

    print("done.");
}
