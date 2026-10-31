// @strict-null false
// An interface value printed, or put into a string: its `toString` when it has one, as a class's is;
// else its fields, read through it, as an object literal's are; null as null. It crashed at run time,
// whatever object was behind it. A field of its own type again prints `[Object]`.
interface Thing {
    n: number;
    name: string;
}

interface Named {
    label: string;
    toString(): string;
}

interface Base {
    a: number;
}

interface Derived extends Base {
    b: string;
}

interface Chain {
    v: number;
    next: Chain;
}

class Ball implements Thing {
    constructor(public n: number, public name: string) {}
}

@value
interface Hit {
    readonly thing: Thing;
    readonly dist: number;
}

function find(d: number): Hit | null {
    if (d < 0) return null;
    return { thing: new Ball(1, "ball"), dist: d };
}

function main() {
    const t: Thing = new Ball(3, "b");
    assert(`${t}` == "{ n: 3, name: 'b' }", "a class seen through an interface");

    const lit: Thing = { n: 4, name: "lit" };
    assert(`${lit}` == "{ n: 4, name: 'lit' }", "an object literal seen through an interface");

    const none: Thing = null;
    assert(`${none}` == "null");

    const named: Named = {
        label: "x",
        toString() {
            return "Named(" + this.label + ")";
        }
    };
    assert(`${named}` == "Named(x)", "its toString");

    const base: Base = { a: 1 };
    const derived: Derived = { a: base.a, b: "two" };
    assert(`${derived}` == "{ a: 1, b: 'two' }", "the fields of what it extends first");

    const chain: Chain = { v: 1, next: { v: 2, next: null } };
    assert(`${chain}` == "{ v: 1, next: [Object] }");

    assert(`${find(2)}` == "{ thing: { n: 1, name: 'ball' }, dist: 2 }", "a @value value holding an interface");
    assert(`${find(-1)}` == "null");

    print(t, lit, named);
    print("done.");
}
