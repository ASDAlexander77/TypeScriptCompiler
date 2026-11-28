// TSLANG 222: language gaps (2026-11-28).
// - A conditional type alias naming itself in a branch (`Flat<T> = T extends (infer U)[] ? Flat<U> : T`) was "circularly
//   references itself", and the inner level's `infer U` found the outer level's U bound (Flat<number[][]> was
//   number[]); `number extends { v: infer U }` was true (a field the type has not got matched the `infer`).
// - `{ x: string } & T` of a generic function was `never` ("'Never' can't be used as parameter type"); a call infers
//   T from its argument.
// - `s.toString()` and `s.description` of a symbol.
// - `Object.assign` into a literal (`Object.assign({}, a, { b: 1 })`) or into a value that has every field the sources
//   give; `Object.is`.
// - `const t = { a: 1, b: 2 }; t.b = 3`: a literal of constants was folded into a constant with no field to write.

type Flat<T> = T extends (infer U)[] ? Flat<U> : T;
type Unbox<T> = T extends { v: infer U } ? Unbox<U> : T;

function flatFirst<T>(x: T[][]): Flat<T[][]> {
    return <any>x[0][0];
}

function getX<T>(o: { x: string } & T): string {
    return o.x;
}

function bump<T>(o: T & { n: number }) {
    return o.n + 1;
}

class P {
    x = 1;
    y = "a";
}

function counter() {
    const c = { hits: 0, misses: 0 };
    for (let i = 0; i < 5; i++) {
        if (i % 2 == 0) c.hits++;
        else c.misses += 1;
    }

    return c.hits * 10 + c.misses;
}

function main() {
    // recursive conditional types
    const a: Flat<string[][][]> = "s";
    const b: Unbox<{ v: { v: number } }> = 2.5;
    const c: Unbox<number> = 3;
    const d: Flat<boolean> = true;
    assert(a == "s" && b == 2.5 && c == 3 && d, "recursive conditional types");
    assert(flatFirst([[7]]) == 7, "Flat of a generic");

    // a generic intersection
    assert(getX({ x: "b", y: 2 }) == "b", "{ x } & T inferred");
    assert(getX<{ y: number }>({ x: "c", y: 1 }) == "c", "{ x } & T given");
    assert(bump({ n: 4, s: "z" }) == 5, "T & { n }");

    // symbols
    const s = Symbol("k");
    assert(s.toString() == "Symbol(k)" && s.description == "k", "symbol members");
    const e = Symbol();
    assert(e.toString() == "Symbol()" && e.description == "", "symbol with no description");
    assert(Symbol.iterator.description == "Symbol.iterator", "well-known symbol");

    // Object.assign
    const t = { a: 1, b: 2 };
    const r = Object.assign(t, { b: 3 });
    assert(r.a == 1 && r.b == 3 && t.b == 3, "assign into a literal's variable");
    const src = { b: 7 };
    Object.assign(t, src);
    assert(t.b == 7, "assign from a variable");
    const u = Object.assign({}, { x: 1 }, { y: "s" });
    assert(u.x == 1 && u.y == "s", "assign into a literal");
    const v = Object.assign({ k: true }, src);
    assert(v.k && v.b == 7, "assign into a literal with fields");
    const p = new P();
    Object.assign(p, { y: "z" }, { x: 5 });
    assert(p.x == 5 && p.y == "z", "assign into a class instance");

    // Object.is
    const nan: number = NaN;
    const zero: number = 0;
    const negativeZero: number = -zero;
    assert(Object.is(nan, nan) && !Object.is(zero, negativeZero) && Object.is(zero, zero), "Object.is of numbers");
    assert(Object.is("a1", "a" + 1) && !Object.is(true, false), "Object.is of others");

    // a const literal whose fields are written
    const w = { a: 1, b: 2 };
    w.b = 3;
    w.a += 10;
    assert(w.a == 11 && w.b == 3, "a const literal's fields written");
    assert(counter() == 32, "a const literal counting");

    print("done.");
}
