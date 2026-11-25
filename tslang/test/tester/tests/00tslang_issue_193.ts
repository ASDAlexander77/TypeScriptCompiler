// Bugs found by the tsc tests sweep (TSLANG 193):
// - a static and an instance field of one name (`static p = 3; p = 4;`, also named by a literal or a symbol): the
//   class read its instance field ("failed to legalize operation 'ts.ClassRef'")
// - a static class as a value (`{ s: Helpers }`, `export default Array`): it has no `..rtti` global ("failed to
//   legalize operation 'ts.AddressOf'")
const k3 = Symbol();
const k4 = "prop" as const;

class Foo {
    static p1 = 3;
    p1 = 4;

    static [k3] = 5;
    [k3] = 6;

    static [k4] = 7;
    [k4] = 8;
}

static class Helpers {
    static twice(n: number) {
        return n * 2;
    }
}

// - object literals given for `Int<T, U>` and `Int<U, T>`: U, given by the first one's method, types the second's
//   parameter (it was the type parameter U, and the call failed with no message)
interface Int<T, U> {
    method(x: T): U;
}

function roundTrip<T, U>(x: T, y: Int<T, U>, z: Int<U, T>): T {
    return z.method(y.method(x));
}

// - an object literal with a method given to an object type with a method signature (`{ method(x: string): number }`):
//   the method's parameter took the signature's `this` (Opaque); and to a union of such shapes told apart by a literal
//   field (`{ kind: "a", method(a) {...} }`), typed by the member it names ("can't be stored")
type Shape = { kind: "a"; method(x: string): number } | { kind: "b"; method(x: number): string };
type One = { kind: "a"; method(x: string): number };

function invoke(item: Shape) {
    return item.kind === "a" ? "" + item.method("12") : item.method(42);
}

// - an array literal given to a union with a one-element tuple in it (`["a"] | I`): "can't be stored"
interface HasOther {
    someOtherProperty: number;
}

function tupleOrInterface(args: ["a"] | HasOther) {
    return "ok";
}

// - a negated test narrows (`!(typeof d === "array")`, `!(d instanceof F)`; `!Array.isArray(d)` with the default library):
//   neither branch was narrowed
class Leaf {
    v = 1;
}

class Branch {
    w = 2;
}

function notArray(d: Leaf | Leaf[]) {
    if (!(typeof d === "array")) {
        return d.v;
    } else {
        return d.length;
    }
}

function notLeaf(d: Leaf | Branch) {
    if (!(d instanceof Leaf)) {
        return d.w;
    }

    return d.v;
}

// - an ambient destructuring declaration (`declare var { a, ...rest } : AB;`, in a declared namespace too): "failed
//   statement" alone; each name is declared as `declare var a: AB["a"]` would be
interface AB {
    a: number;
    b: string;
}

declare var { a: ambientA, ...ambientRest }: AB;

declare namespace Ambient {
    export var { b }: AB;
}

// - `[n, ...rest] = robot` of a tuple: the rest took the last element cast to an array
type Robot = [number, string, string];

// read in module variables, as the sweep's tests did
const t7 = Foo[k3];
const t8 = new Foo()[k3];
const holder = { s: Helpers };

function main() {
    assert(Foo.p1 == 3 && new Foo().p1 == 4 && Foo["p1"] == 3, "a static and an instance field of one name");
    assert(Foo[k4] == 7 && new Foo()[k4] == 8, "fields named by a literal");
    assert(t7 == 5 && t8 == 6 && Foo[k3] == 5 && new Foo()[k3] == 6, "fields named by a symbol");

    const local = { s: Helpers };
    assert(local.s.twice(2) == 4 && holder.s.twice(3) == 6, "a static class as a value");

    const r = roundTrip("abc", { method(p1) { return p1.length } }, { method(p2) { return "n" + p2 } });
    assert(r == "n3", "U inferred from one object literal's method types the next one's");

    const one: One = { kind: "a", method(a) { return a.length; } };
    assert(one.method("123") == 3, "a method typed by a method signature");
    assert(invoke({ kind: "a", method(a) { return a.length; } }) == "2", "a literal typed by the member its kind names");
    assert(invoke({ kind: "b", method(n) { return "n" + n; } }) == "n42", "a literal typed by the other member");

    assert(tupleOrInterface(["a"]) == "ok", "an array literal given to a union with a tuple");

    assert(notArray(new Leaf()) == 1 && notArray([new Leaf(), new Leaf()]) == 2, "!(typeof d === \"array\") narrows");
    assert(notLeaf(new Branch()) == 2 && notLeaf(new Leaf()) == 1, "!(d instanceof Leaf) narrows");

    const robot: Robot = [1, "mower", "mowing"];
    let n: number;
    let rest: (number | string)[];
    [n, ...rest] = robot;
    assert(n == 1 && rest.length == 2 && rest[0] == "mower" && rest[1] == "mowing", "the rest of a tuple assigned");

    print("done.");
}
