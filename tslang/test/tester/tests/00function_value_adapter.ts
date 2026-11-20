// A function value read at run time - a parameter, a variable - given as a function type it is adapted to: it goes
// into a box, and an adapter bound to the box calls it. Fewer parameters ignore the rest of the arguments, an optional
// extra one is given `undefined`, a result is dropped (`void`) or widened (`string` as `string | number`). Declared
// unions of function types are one merged type, so a value is given as it this way (`let u: H = cb`).
// Not under -mm=own yet: a function value in a box takes a second reference to it.
let count: number = 0;

type H = (() => void) | (() => number);
type HR = (() => string) | (() => number);
type HO = ((v: number) => void) | ((v: number, i?: number) => void);

function reg(cb: () => number) {
    let u: H = cb;
    u();
}

function widen(cb: () => string): number {
    let u: HR = cb;
    const r = u();
    return typeof r == "string" ? r.length : -1;
}

function more(cb: (a: number) => void) {
    const f: (a: number, b: number) => void = cb;
    f(5, 6);
}

function main() {
    reg(() => { count += 1; return 1; });
    assert(count == 1);

    let k: number = 0;
    for (let i = 0; i < 100; i++) {
        k += widen(() => "ab" + i);
    }
    assert(k == 390);

    more((a: number) => { count += a; });
    assert(count == 6);

    let ho: HO = (x: number, i?: number) => { count += i === undefined ? 100 : 1000; };
    let cb1: (value: number) => void = ho;
    cb1(10);
    assert(count == 106);

    print("done.");
}
