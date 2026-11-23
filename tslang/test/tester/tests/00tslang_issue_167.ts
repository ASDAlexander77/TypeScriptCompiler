// TSLANG_ISSUES 167: narrowing an optional union (`typeof x === "number"` of an `x?: number | string`) takes the
// union out of the optional as it is (it warned "casting from optional (undefined) value", with no location); a
// type guard is a boolean to whoever calls it as a function value (`xs.filter(isDefined)`, `const p: (v) => boolean
// = isDef`), and a function taking a number can be given where an integer is passed (`[1, -2, 3].filter(pos)`).
class A {
    n = 1;
}

class B {
    s = "b";
}

function f(x?: number | string) {
    if (typeof x === "number") return x + 1;
    if (typeof x === "string") return x.length;
    return 0;
}

function g(v?: A | B) {
    if (v instanceof A) return v.n;
    if (v instanceof B) return v.s.length;
    return -1;
}

function eq(x?: number | string) {
    return x == "a";
}

function pos(v: number): boolean {
    return v > 0;
}

function isDef(v: number | undefined): v is number {
    return v !== undefined;
}

function isDefined<T>(v: T | undefined): v is T {
    return v !== undefined;
}

function main() {
    assert(f(1) == 2 && f("ab") == 2 && f() == 0, "typeof of an optional union");
    assert(g(new A()) == 1 && g(new B()) == 1 && g() == -1, "instanceof of an optional union");
    assert(eq("a") && !eq(1) && !eq(), "== of an optional union");

    let count: number = 0;
    for (const v of [1, -2, 3].filter(pos)) count++;
    assert(count == 2, "a number parameter given integers");

    const xs: (number | undefined)[] = [1, undefined, 3];
    count = 0;
    for (const v of xs.filter(isDef)) count++;
    assert(count == 2, "a type guard as a callback");

    count = 0;
    for (const v of xs.filter(isDefined)) count++;
    assert(count == 2, "a generic type guard as a callback");

    const p: (v: number | undefined) => boolean = isDef;
    assert(!p(undefined) && p(1), "a type guard as a function returning boolean");
    print("done.");
}
