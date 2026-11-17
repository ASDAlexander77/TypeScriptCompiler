// A tuple is an array as well: an array literal of elements of different types (`[1, "a"]`) has
// the members of Array that read it, taken from an array of its elements of the union of their
// types. A class named as an element is its value (`[A, B]` holds `typeof A | typeof B`), which
// `new` makes whichever class it holds of (abstractClassUnionInstantiation.ts).
// map and filter give a generator without the default library: what they give is counted by for...of.
class ConcreteA {
    a = 1;
}

class ConcreteB {
    b = 2;
}

class ConcreteC extends ConcreteA {
}

type Concretes = typeof ConcreteA | typeof ConcreteB;

function make(c: Concretes) {
    return new c();
}

class P {
    n: number;
    constructor(n: number) {
        this.n = n;
    }
}

class Q {
    n: number;
    constructor(n: number) {
        this.n = n;
    }
}

function makeWith(c: typeof P | typeof Q, n: number) {
    return new c(n);
}

function makeWithF(c: typeof P | typeof Q, f: (x: number) => number) {
    return new c(f(1));
}

function main() {
    // a literal of mixed types
    let flags = 0;
    let all = 0;
    for (const k of [1, true, 2].map(x => typeof x == "boolean" ? 1 : 0)) {
        flags += k;
        all++;
    }

    assert(all == 3 && flags == 1, "map of a mixed literal");

    let booleans = 0;
    for (const b of [1, true, 2, false].filter(x => typeof x == "boolean")) {
        booleans++;
    }

    assert(booleans == 2, "filter of a mixed literal");

    // a tuple read from where it is kept, and its length
    let t: [number, boolean] = [3, true];
    assert(t.length == 2, "length of a tuple");
    let mapped = 0;
    for (const m of t.map(x => 1)) {
        mapped += m;
    }

    assert(mapped == 2, "map of a tuple");
    assert([true, 1, "x"].length == 3, "length of a mixed literal");

    // classes named in an array literal are their values: `new` makes the class each holds
    let index = 0;
    for (const o of [ConcreteA, ConcreteB].map(cls => new cls())) {
        if (index == 0) {
            assert(o instanceof ConcreteA && !(o instanceof ConcreteB), "first is ConcreteA");
        } else {
            assert(o instanceof ConcreteB && !(o instanceof ConcreteA), "second is ConcreteB");
        }

        index++;
    }

    assert(index == 2, "made both");

    // `new` through a value of a union of class types, given each class in turn
    const b = make(ConcreteB);
    assert(b instanceof ConcreteB && !(b instanceof ConcreteA), "make(ConcreteB)");
    const a = make(ConcreteA);
    assert(a instanceof ConcreteA && !(a instanceof ConcreteB), "make(ConcreteA)");
    // a subclass given where its base's value is a member
    const c = make(ConcreteC);
    assert(c instanceof ConcreteC && c instanceof ConcreteA, "make(ConcreteC)");

    // three of them, the dispatch over more than two members
    let count = 0;
    for (const o of [ConcreteA, ConcreteB, ConcreteC].map(cls => new cls())) {
        if (o instanceof ConcreteA) count++;
    }

    assert(count == 2, "ConcreteA and ConcreteC");

    // the arguments run once, whichever class is made; a function among them is made once
    let i = 10;
    const p = makeWith(Q, i++);
    assert(p instanceof Q && p.n == 10 && i == 11, "argument run once");
    const k = 5;
    const q = makeWithF(P, (x: number) => x + k);
    assert(q instanceof P && q.n == 6, "function argument");

    print("done.");
}
