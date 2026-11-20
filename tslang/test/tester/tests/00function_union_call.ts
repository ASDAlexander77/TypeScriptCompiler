// A declared union of function types (`(() => void) | (() => number)`) is one type it can be called as: the
// widest parameters, returning `void` when one of them does. A function returning a value is given as one returning
// `void` through a thunk that drops (releases) the result, as TypeScript allows (`let f: () => void = n`).
let count: number = 0;

type handler4 = () => void;
type handler5 = handler4 | (() => number);
type H2 = ((a: number) => void) | (() => string);

function n(): number { count += 1; return 2; }
function v() { count += 10; }
function s(): string { count += 100; return "a" + count; }
function w(a: number) { count += a; }

function pick(k: boolean): handler5 {
    return k ? n : v;
}

function main() {
    let x: handler5 = n;
    x();
    x = v;
    x();
    assert(count == 11);

    let y: H2 = s;
    y(1000);
    y = w;
    y(1000);
    assert(count == 1111);

    pick(true)();
    pick(false)();
    assert(count == 1122);

    let f: () => void = s;
    for (let i = 0; i < 10; i++) {
        f();
    }
    assert(count == 2122);

    let k = 1;
    let h: () => void = () => { k++; return "x" + k; };
    h();
    assert(k == 2);

    print("done.");
}
