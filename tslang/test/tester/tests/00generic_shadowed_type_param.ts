// A generic class instantiated inside a generic function with a type parameter of the same name as the class's
// own (`Box<S>` in `f<T, S>`, Box's parameter named T): the class's own shadows the function's. Before, `Box<S>`
// was made as `Box<T>`: `f(Box<number>, Box<string>)` inferred T = number | string and S = any, and read its
// second argument with the wrong layout (a crash).
class Box<T> {
    constructor(public v: T) {}
}

interface Holder<T> {
    held: T;
}

type Pair<T> = { first: T, second: T };

function f<T, S>(a: Box<T>, b: Box<S>) {
    return typeof a.v + " " + typeof b.v + " " + b.v;
}

function g<T, S>(a: Holder<T>, b: Holder<S>) {
    return typeof a.held + " " + typeof b.held;
}

function h<T, S>(a: Pair<T>, b: Pair<S>) {
    return typeof a.first + " " + typeof b.second;
}

function main() {
    assert(f(new Box<number>(1), new Box<string>("x")) == "number string x", "class");
    assert(g({ held: 1.5 }, { held: "y" }) == "number string", "interface");
    assert(h({ first: 1.5, second: 2.5 }, { first: "a", second: "b" }) == "number string", "type alias");
    print("done.");
}
