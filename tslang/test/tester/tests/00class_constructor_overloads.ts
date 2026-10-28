// Constructor overloads: a class may implement several constructors, and `new` picks one by its
// arguments. The parameter properties of one constructor are not assigned in another (`this.data =
// data` in `constructor(count: int)` was "can't resolve name: data"), and a tie - `new A()` of two
// constructors whose parameters all have defaults - goes to the one declared first.

class A<T> {
    constructor(private data: T[] = []) {
    }

    constructor(count: int = 0) {
        this.data = [];
        this.data.length = count;
    }

    get length() {
        return this.data.length;
    }
}

class B {
    private s: string;
    private n = 7;

    constructor(s: string = "x") {
        this.s = s;
    }

    constructor(count: int, extra: string) {
        this.s = extra;
        this.n = count;
    }

    get v() {
        return this.s;
    }

    get k() {
        return this.n;
    }
}

// an optional parameter fits as the type it holds: `(count: int = 0)` takes an int as well as the
// rest parameter's elements do, and declared first it wins (it was scored as optional<int>, by
// conversion, and `new C<int>(10)` went to `(...args: T[])`)
class C<T> {
    n = 0;

    constructor(count: int = 0) {
        this.n = count;
    }

    constructor(...args: T[]) {
        this.n = -args.length;
    }
}

function main() {
    const c1 = new C<int>(10);
    const c3 = new C<int>(1, 2, 3);
    assert(c1.n == 10);
    assert(c3.n == -3);

    const a0 = new A<int>();
    const a1 = new A<int>(10);
    const a2 = new A<int>([1, 2]);
    assert(a0.length == 0);
    assert(a1.length == 10);
    assert(a2.length == 2);

    const b0 = new B();
    const b1 = new B("q");
    const b2 = new B(3, "e");
    assert(b0.v == "x" && b0.k == 7);
    assert(b1.v == "q" && b1.k == 7);
    assert(b2.v == "e" && b2.k == 3);

    print("done.");
}
