// A generic class written without type arguments takes them from its constructor's arguments, as
// a generic function's call does. `new Set([2, 4, 6, 8])` was Set<any> (its default), and a class
// without defaults was "missing type arguments".
class Box<V> {
    first: V;
    count: number;
    constructor(values: V[]) {
        this.first = values[0];
        this.count = values.length;
    }
}

class Defaulted<V = any> {
    value: V;
    constructor(value?: V) {
        this.value = value;
    }
}

class Pairs<K = any, V = any> {
    firstKey: K;
    firstValue: V;
    constructor(entries?: [K, V][]) {
        this.firstKey = entries[0][0];
        this.firstValue = entries[0][1];
    }
}

class Tagged<T, Tag = string> {
    tag: Tag;
    constructor(public item: T, tag?: Tag) {
        this.tag = tag;
    }
}

class Applied<T> {
    result: T;
    constructor(start: T, step: (x: T) => T) {
        this.result = step(start);
    }
}

function boxOf<T>(items: T[]) {
    return new Box(items);
}

// the same inference a function's call makes: an unnamed tuple's fields are matched by position
// (both took the first element's type, and this did not compile)
function firstValue<K, V>(entries: [K, V][]): V {
    return entries[0][1];
}

function main() {
    // no default: inferred, or it would not compile
    const numbers = new Box([2, 4, 6, 8]);
    assert(numbers.first + 1 == 3);
    assert(numbers.count == 4);

    const strings = new Box(["a", "b"]);
    assert(strings.first + "!" == "a!");

    // with a default: inferred, not the default
    const seven = new Defaulted(7);
    assert(seven.value * 2 == 14);

    // no arguments: the default, as before
    const nothing = new Defaulted();
    assert(nothing != null);

    // the declared type wins over the arguments
    const declared: Defaulted<number> = new Defaulted(3);
    assert(declared.value + 0.5 == 3.5);

    // both parameters from a tuple element type
    const pairs = new Pairs([["one", 1], ["two", 2]]);
    assert(pairs.firstKey + "?" == "one?");
    assert(pairs.firstValue * 10 == 10);

    // the second parameter is not reached and keeps its default
    const tagged = new Tagged(5);
    assert(tagged.item * 3 == 15);
    const taggedToo = new Tagged(5, "five");
    assert(taggedToo.tag == "five");

    // a callback's parameters come from the constructor's: T is taken from `start` alone
    const applied = new Applied(20, (x) => x + 1);
    assert(applied.result == 21);

    // inside a generic function: T is known once boxOf is specialized
    const boxed = boxOf([1.5, 2.5]);
    assert(boxed.first * 2 == 3);

    assert(firstValue([["one", 1], ["two", 2]]) * 10 == 10);

    print("done.");
}
