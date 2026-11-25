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

    print("done.");
}
