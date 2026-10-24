// `@value class`: `new` makes a value - a struct of the class's readonly fields on the stack,
// with no box and no vtable. Static helpers, an instance method and a getter (both take `this`
// by value), a static field of the class's own type and one of another class; values passed,
// returned, stored in a field of an ordinary class and in an array, and copied. A constructor passes
// `this` on as a value and calls a method on it.
@value
class V {
    constructor(public readonly x: number, public readonly y: number, public readonly z: number) {}

    static plus(a: V, b: V) {
        return new V(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    static white = new V(1, 2, 3);

    dot(o: V) {
        return this.x * o.x + this.y * o.y + this.z * o.z;
    }

    get len2() {
        return this.dot(this);
    }
}

// fields assigned in the constructor
@value
class P {
    readonly a: number;
    readonly b: number;

    constructor(a: number) {
        this.a = a;
        this.b = a * 2;
    }
}

// a constructor seeing `this` as a value: passed to a function (a copy of what is set so far) and
// a method called on it
function sumOf(q: Q) {
    return q.a + q.b;
}

@value
class Q {
    readonly a: number;
    readonly b: number;
    readonly s: number;
    readonly t: number;

    constructor(a: number) {
        this.a = a;
        this.b = a * 2;
        this.s = sumOf(this);
        this.t = this.twice();
    }

    twice() {
        return this.a * 2;
    }
}

// a static field of an ordinary class holding a value
class K {
    static k = new V(9, 0, 0);
}

// an ordinary class holding a value
class Cam {
    pos: V;
    n = 0;

    constructor(p: V) {
        this.pos = p;
    }
}

function main() {
    assert(V.plus(new V(1, 2, 3), new V(4, 5, 6)).y == 7);
    assert(new V(1, 2, 2).len2 == 9);
    assert(V.white.y == 2);
    assert(V.white.x == 1 && V.white.z == 3);
    assert(K.k.x == 9);
    assert(new Q(3).s == 9);
    assert(new Q(3).t == 6);
    assert(new P(3).b == 6);

    // the value keeps its own fields after the class's other fields change
    const c = new Cam(new V(1, 1, 1));
    c.n = 5;
    c.pos = V.plus(c.pos, c.pos);
    assert(c.pos.x == 2);

    const arr: V[] = [new V(1, 0, 0), new V(2, 0, 0)];
    assert(arr[1].x == 2);

    // a copy, not a reference
    let v = new V(1, 0, 0);
    const w = v;
    v = new V(9, 0, 0);
    assert(w.x == 1);

    let acc = new V(0, 0, 0);
    for (let i = 0; i < 10000; i++) {
        acc = V.plus(acc, new V(1, 0, 0));
    }

    assert(acc.x == 10000);

    print("done.");
}
