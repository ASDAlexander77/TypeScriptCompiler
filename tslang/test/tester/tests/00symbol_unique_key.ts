// `Symbol("d")` is a unique symbol known at compile time (user ruling 11-18): a `const` holding one is a
// constant key, as a `const` holding a string literal is, and at run time a symbol equals only itself
const k = Symbol("k");
const k2 = Symbol("k");
const forA = Symbol.for("a");

class Foo {
    [k](): string {
        return "method";
    }

    static [k2](): number {
        return 2;
    }
}

function kind(x: symbol | string) {
    switch (x) {
        case k:
            return "k";
    }

    return "other";
}

function localKey() {
    const local = Symbol();
    const o = { [local]: 5 };
    return o[local];
}

function typeOfAny(a: any) {
    return typeof a;
}

function main() {
    const obj = { a: 1, [k]: "by symbol" };
    assert(obj[k] == "by symbol", "object literal key");

    assert(typeof k == "symbol", "typeof");
    assert(typeOfAny(k) == "symbol", "typeof in any");
    assert(k === k, "same symbol");
    assert(k !== k2, "same description, another symbol");
    assert(forA === Symbol.for("a"), "Symbol.for of one description");

    assert(new Foo()[k]() == "method", "method key");
    assert(Foo[k2]() == 2, "static method key");

    assert(localKey() == 5, "local key");

    let s: symbol = k;
    assert(s === k && s !== k2, "symbol variable");

    assert(kind(k) == "k" && kind(k2) == "other" && kind("k") == "other", "switch of a union");

    const symbols: symbol[] = [k, Symbol.iterator];
    assert(symbols[0] === k && symbols[1] !== k, "array of symbols");

    assert(String(k) == "Symbol(k)", "text of a symbol");
    assert(String(Symbol.iterator) == "Symbol(Symbol.iterator)", "text of a well-known symbol");
    assert(String(Symbol.iterator).length == 23, "length of the text of a symbol");

    print("done.");
}
