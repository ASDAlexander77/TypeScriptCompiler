// @strict-null false
// Without strict null checks a nullable type (string, a class) absorbs null: its pointer holds it. A
// union with a nullable member is no pointer, though, so there null has to stay a member of its own
class C {
    x = 1;
}

function isNullUnion(p: string | number | null) {
    return p === null;
}

function unionWithNullableMember() {
    let u: string | number | null = null;
    assert(u === null, "union: null is null");

    u = "a";
    assert(u === "a", "union: string");

    u = 1;
    assert(u === 1, "union: number");

    assert(isNullUnion(null), "union argument: null");
    assert(!isNullUnion("b"), "union argument: string");
    assert(!isNullUnion(2), "union argument: number");

    let v: C | number | null = null;
    assert(v === null, "class union: null is null");
    v = 3;
    assert(v !== null, "class union: number is not null");
}

function nullableOnly() {
    let s: string | null = null;
    assert(s === null, "string: null");

    let o: string | null | undefined = null;
    assert(o === null, "optional string: null");
    o = undefined;
    assert(o === undefined, "optional string: undefined");

    let c: C | null | undefined = null;
    assert(c === null, "optional class: null");

    let n: number | null | undefined = null;
    assert(n === null, "optional number: null");
    n = undefined;
    assert(n === undefined, "optional number: undefined");
}

function main() {
    unionWithNullableMember();
    nullableOnly();
    print("done.");
}
