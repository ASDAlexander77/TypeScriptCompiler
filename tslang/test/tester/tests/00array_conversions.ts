// Arrays given where another kind of array is expected:
// - concat's arguments that are not arrays are one element each (`a.concat("x", ["y"])`)
// - `{}` given as an array is an empty array
// - an array given as `any[]` is a copy with each element boxed
// - an `Array<T>` class holding its elements in `data` (the default library's) and a `T[]` go into
//   each other: the class gives its `data`, and an array becomes a new class holding it
class Array<T> {
    constructor(protected data: T[] = []) {
    }

    get length() { return this.data.length; }
}

// as the default library declares it (lib.generics.ts): arrays only
namespace __Array {
    function concat<T>(this: T[], ...other: T[][]) {
        let newArray: T[] = [];
        for (const v of this) newArray.push(v);
        for (const item of other) {
            for (const v of item) newArray.push(v);
        }

        return newArray;
    }
}

type L = "foo" | "bar";

function count(a: any[]) { return a.length; }
function firstOf(a: any[]) { return a[0]; }
function size(a: string[]) { return a.length; }
function empty(a: string[], n: number) { return a.length + n; }

function main() {
    const a: string[] = ["x"];
    const r = a.concat("hello", ["world", "!"]);
    assert(r.length == 4);
    assert(r[1] == "hello" && r[3] == "!");

    assert(empty({}, 2) == 2);

    const l: L[] = ["bar", "foo"];
    assert(count(l) == 2);
    assert(typeof firstOf(l) == "string");

    const c: Array<string> = ["p", "q", "r"];
    assert(c.length == 3);
    assert(size(c) == 3);

    print("done.");
}
