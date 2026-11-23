// TSLANG 175: a getter that reads itself ("Class member 'p' can't be found"); an interface of one call signature takes
// a function ("invalid cast from (p0: number) => number to Cb"); an interface extending a generic interface beside a
// function of its name ("interface cannot extend type"); `for (var x in this)` in a method walks the fields

class Counter {
    n = 3;
    get p(): number {
        return this.n > 0 ? this.n-- + this.p : 0;
    }
}

class A {
    get v(): number {
        return 1;
    }
    show() {
        return this.v;
    }
}

class B extends A {
    get v(): number {
        return 2;
    }
}

interface Cb extends Function {
    (x: number): number;
}

function apply(c: Cb) {
    return c(3);
}

interface Tag<Self, T> {
    t: T;
}

function Tag<T>(x: T): T {
    return x;
}

interface K extends Tag<K, number> {
    key: number;
}

class Base {
    a = 1;
}

class Fields extends Base {
    b = "x";
    c = true;
    keys() {
        let s = "";
        for (var x in this) {
            s = s + x + ",";
        }
        return s;
    }
    m() {
        return 1;
    }
}

function main() {
    assert(new Counter().p == 6);

    const a: A = new B();
    assert(a.v == 2 && a.show() == 2);

    assert(apply((x: number) => x * 2) == 6);
    const c: Cb = (x: number) => x + 1;
    assert(c(1) == 2);

    const k: K = { t: 1, key: 2 };
    assert(k.key == 2 && k.t == 1);

    assert(new Fields().keys() == "a,b,c,");

    print("done.");
}
