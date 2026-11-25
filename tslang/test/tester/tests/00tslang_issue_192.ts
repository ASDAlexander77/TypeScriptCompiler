// Bugs found by the tsc tests sweep (TSLANG 192):
// - `typeof` narrows a field (`typeof this._x === "number" ? this._x : this._x.bar`), in a ternary and an `if`
// - a getter and a setter of different types, read and written by `+=`
// - `instanceof` on a union of an interface and classes
// - `new C()` in a module variable above `class C` ("failed to legalize operation 'ts.AddressOf'")
// - a class given as a parameter's default (`zz = C`, a static method's `zz = this`)
type Foo = { bar: number };

class NumberOrObject {
    _x: number | Foo = 0;
    get x(): number {
        return typeof this._x === "number" ? this._x : this._x.bar;
    }
    set x(value: number | Foo) {
        this._x = value;
    }
    twice(): number {
        if (typeof this._x !== "number") {
            return this._x.bar * 2;
        }

        return this._x * 2;
    }
}

interface Base {
    foo: number;
}

class D1 implements Base {
    foo: number = 1;
}

class D2 implements Base {
    foo: number = 2;
    bar: number = 3;
}

function which(x: Base | D1): number {
    if (x instanceof D2) {
        return x.bar;
    }

    return x.foo;
}

var before: C = new C();

class C {
    n = 3;
    static y = 5;
    static viaThis(zz = this) {
        return zz.y;
    }
    static viaName(zz = C) {
        return zz.y;
    }
}

function main() {
    const o = new NumberOrObject();
    o.x += 4;
    assert(o.x == 4, "a getter and a setter of other types, by +=");
    o.x = { bar: 7 };
    assert(o.x == 7 && o.twice() == 14, "a field narrowed by typeof");

    assert(which(new D2()) == 3 && which(new D1()) == 1, "instanceof on a union of an interface and classes");

    assert(before.n == 3, "a class used above its declaration");

    assert(C.viaThis() == 5 && C.viaName() == 5, "a class as a parameter's default");

    print("done.");
}
