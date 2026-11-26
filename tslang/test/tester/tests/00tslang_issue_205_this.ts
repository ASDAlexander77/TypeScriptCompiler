// TSLANG 205: an arrow in an object literal reads `this` and `super` of the code around it, as in JavaScript (user
// ruling 11-26) - not the literal. And a closure given for an interface's function field through an adapter (an s32
// result for a `number` one) keeps its captures.

class Base {
    m() { return 1; }
}

class Derived extends Base {
    n: number = 5;

    m() { return 2; }

    literal() {
        return { viaSuper: () => super.m(), eager: super.m(), viaThis: () => this.m() + this.n };
    }
}

interface HasF {
    f: () => number;
}

class Holder {
    k = 7;

    viaVariable(): HasF {
        const f = () => this.k;
        return { f: f };
    }

    direct(): HasF {
        return { f: () => this.k };
    }

    typedLocal() {
        const o: HasF = { f: () => this.k };
        return o.f();
    }

    untypedLocal() {
        const o = { g: (x: number) => x + this.k };
        return o.g(1);
    }
}

// in a field's initializer: the instance, and an untyped parameter is not the closure's captures
class WithField {
    k = 3;
    lit = { d: (cb) => cb(this), later: (cb) => () => cb(this) };
}

function capturing(): HasF {
    let k = 9;
    const f = () => k;
    return { f: f };
}

function main() {
    const o = new Derived().literal();
    assert(o.viaSuper() == 1, "super");
    assert(o.eager == 1, "eager super");
    assert(o.viaThis() == 7, "this");

    const h = new Holder();
    assert(h.viaVariable().f() == 7, "via variable");
    assert(h.direct().f() == 7, "direct");
    assert(h.typedLocal() == 7, "typed local");
    assert(h.untypedLocal() == 8, "untyped local");
    assert(capturing().f() == 9, "capturing");

    const w = new WithField();
    assert(w.lit.d((x: WithField) => x.k) == 3, "field d");
    assert(w.lit.later((x: WithField) => x.k + 1)() == 4, "field later");

    // a method of a literal still has the literal as its `this`
    const lit = { v: 3, get() { return this.v; } };
    assert(lit.get() == 3, "method this");

    print("done.");
}
