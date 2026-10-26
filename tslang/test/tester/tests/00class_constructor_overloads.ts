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

function main() {
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
