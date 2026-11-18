// A value of a generic function type holds one function: the type's parameters are erased, each its
// constraint, else `any` (`let b: <T>(x: T) => T`, `declare const p: <T>(fn: () => T) => T`, a parameter
// `f: <U>(u: U) => U`). The variable was refused with nothing said but "failed statement".
interface Named { name: string }

let b: <T>(x: T) => T = (x) => x;
let first: <T extends Named>(items: T[]) => string = (items) => items[0].name;

function apply<T>(f: <U>(u: U) => U, v: T) { return f(v); }

function main() {
    assert(b(5) == 5 && b("s") == "s");
    assert(first([{ name: "n" }]) == "n");
    assert(apply((x) => x, 3) == 3);
    let local: <T>(x: T) => T = (x) => x;
    assert(local(1) == 1);
    print("done.");
}
