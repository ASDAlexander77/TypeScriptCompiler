// A class may have a static and an instance member of the same name, as TypeScript allows: both
// were one function symbol, so the second was "another declaration of the same name", and a call
// through the class found the instance method
interface Comparable {
    compare(other: A): number;
}

class A implements Comparable {
    constructor(public x: number) {}

    public compare(other: A): number {
        return this.x - other.x;
    }

    public static compare(a: A, b: A): number {
        return a.x < b.x ? -1 : a.x > b.x ? 1 : 0;
    }

    // the static one through `this` in a static method, the instance one through `this` in an instance one
    public static sign(a: A, b: A): number {
        return this.compare(a, b);
    }

    public distance(other: A): number {
        const d = this.compare(other);
        return d < 0 ? -d : d;
    }

    // the instance one calls the static one by the class
    public order(other: A): number {
        return A.compare(this, other);
    }

    private static count = 0;

    public get label(): string {
        return "A" + this.x;
    }

    public static get label(): string {
        return "class A";
    }
}

class B {
    // the static one first
    public static make(n: number): B {
        return new B(n);
    }

    constructor(public n: number) {}

    public make(n: number): B {
        return new B(this.n + n);
    }
}

class C extends A {
    constructor(x: number) {
        super(x * 10);
    }
}

class Box<T> {
    constructor(public value: T) {}

    public of(value: T): Box<T> {
        return new Box<T>(value);
    }

    public static of(value: number): Box<number> {
        return new Box<number>(value + 1);
    }
}

function main() {
    const p = new A(1);
    const q = new A(3);
    assert(p.compare(q) == -2, "the instance method");
    assert(A.compare(p, q) == -1 && A.compare(q, p) == 1 && A.compare(p, p) == 0, "the static method");
    assert(A.sign(q, p) == 1, "this in a static method");
    assert(q.distance(p) == 2, "this in an instance method");
    assert(p.order(q) == -1, "the static one from an instance method");
    assert(p.label == "A1" && A.label == "class A", "a static and an instance getter");

    assert(B.make(2).make(5).n == 7, "the static one declared first");

    const c = new C(2);
    assert(c.compare(p) == 19, "inherited instance method");

    const cmp: Comparable = q;
    assert(cmp.compare(p) == 2, "through an interface");

    const k: typeof A = A;
    assert(k.compare(q, p) == 1, "through a class value");

    // (a static method of a generic class is not reached through the class yet, twin or not)
    assert(new Box<number>(5).of(9).value == 9, "a generic class");

    print("done.");
}
