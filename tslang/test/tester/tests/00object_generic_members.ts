// A generic function held by an object literal's field (`{ test: <T>(value: T): T => value }`, `{ m<T>(x: T) {} }`)
// is one function, its type parameters `any`, as its untyped parameters are: "can't find type by name: T".
const o = {
    test: <T>(value: T): T => value,
    method<T>(value: T): T {
        return value;
    },
    withDefault: <T = undefined>(value: T) => value,
};

function main() {
    assert(o.test(5) == 5);
    assert(o.test("s") == "s");
    assert(o.method(7) == 7);
    assert(o.withDefault(9) == 9);
    const local = { id: <U>(u: U): U => u };
    assert(local.id(3) == 3);
    print("done.");
}
