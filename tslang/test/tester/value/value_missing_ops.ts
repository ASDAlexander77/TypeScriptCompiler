// @strict-null false
// A @value type that may be missing (`T | null`, `T | undefined`), in three places that failed:
// - made by `?:` from null and a literal holding a new object: under rc the branch released the
//   object before the result kept it, and the program read freed memory;
// - `??`: its type was `T | (T | null)`, and the program did not compile;
// - printed: undefined printed as null or as a zeroed T, and a null `T | null` whose T holds a class
//   (no tag, MLIRTypeHelper::getValueNullNiche) printed T's fields instead of null.
// Each runs for a T holding a class (no tag) and a T of numbers only (tagged).
class Owner {
    constructor(public id: number) {}

    toString() {
        return "Owner" + this.id;
    }
}

@value
interface Mark {
    readonly at: number;
    readonly owner: Owner;
}

@value
interface Num {
    readonly at: number;
}

function churn() {
    const junk: Owner[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Owner(999));
}

function mark(d: number): Mark | null {
    return d < 0 ? null : { at: d, owner: new Owner(d) };
}

function num(d: number): Num | null {
    return d < 0 ? null : { at: d };
}

function main() {
    // ?: of null and a literal holding a new object
    const m = mark(4);
    churn();
    assert(m.owner.id == 4 && m.at == 4, "the object the literal holds outlives the branch");
    assert(mark(-1) === null && num(-1) === null && num(3).at == 3);
    let total: number = 0;
    for (let i = 0; i < 1000; i++) {
        const k = mark(i % 3 - 1);
        if (k != null) total += k.owner.id;
    }
    assert(total == 333, "made in a loop");

    // ??
    assert((mark(-1) ?? mark(3)).owner.id == 3 && (mark(2) ?? mark(3)).owner.id == 2);
    const fallback: Mark = { at: 7, owner: new Owner(7) };
    assert((mark(-1) ?? fallback).at == 7);
    assert((num(-1) ?? num(4)).at == 4 && (num(5) ?? { at: 6 }).at == 5);
    const nothing: Mark | undefined = undefined;
    assert((nothing ?? fallback).at == 7);

    // printed
    assert(`${mark(-1)}` == "null", "null of a T holding a class");
    assert(`${mark(2)}` == "{ at: 2, owner: Owner2 }");
    assert(`${num(-1)}` == "null" && `${num(5)}` == "{ at: 5 }");
    const u: Mark | undefined = undefined;
    const v: Num | undefined = undefined;
    const n: Mark | null | undefined = null;
    assert(`${u}` == "undefined" && `${v}` == "undefined" && `${n}` == "null", "undefined and null apart");
    print(mark(-1), mark(2), u, v);

    print("done.");
}
