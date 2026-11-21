// `this` in a parameter's default value, in an arrow there too (`bar(f = () => this)`): "can't resolve name: this",
// then a closure held in a plain function type that lost its captures. A function given as a default is held as a
// hybrid function. And `static y = this` is the class's value, as `static self = P` is (it failed in lowering).
class P {
    static k = 7;
    static y = this;
    x = this;
    n = 3;

    foo(zz = this) {
        return zz.n;
    }

    bar(f = () => this) {
        return f().n;
    }

    baz(g = () => this.n * 2) {
        return g();
    }

    constructor(public z = 1, zzz = (p = this) => this) {
        this.n = zzz().n + 1;
    }
}

function main() {
    const p = new P();
    assert(p.n == 4);
    assert(p.foo() == 4);
    assert(p.bar() == 4);
    assert(p.baz() == 8);
    assert(p.baz(() => 1) == 1);
    assert(p.x.n == 4);
    assert(P.y.k == 7);
    print("done.");
}
