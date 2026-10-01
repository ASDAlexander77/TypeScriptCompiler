// A union equals a value when the member it holds does: a member that cannot be compared with the
// value - null, a class against a number - is simply not equal to it. Cast to the value's type, the
// union read null as a number, a number as an s32 (2.5 === 2), and a class did not compile
class C {
    x = 1;
}

function nullableNumber() {
    let n: number | null = 2.5;
    assert(!(n === 2), "2.5 === 2");
    assert(n === 2.5, "2.5 === 2.5");
    assert(n !== 2, "2.5 !== 2");
    assert(2.5 === n, "2.5 === 2.5, union on the right");
    assert(n == 2.5, "2.5 == 2.5");

    let zero: number = 0;
    n = null;
    assert(!(n === 0), "null === 0");
    assert(n !== 0, "null !== 0");
    assert(!(n == 0), "null == 0");
    assert(!(n === zero), "null === zero");
    assert(n === null, "null === null");

    n = 0;
    assert(n === 0, "0 === 0");
    assert(n === zero, "0 === zero");
}

function classOrNumber() {
    let v: C | number = 3;
    assert(v === 3, "3 === 3");
    assert(!(v === 4), "3 === 4");
    assert(3 === v, "3 === 3, union on the right");

    let c = new C();
    v = c;
    assert(!(v === 3), "class === 3");
    assert(v !== 3, "class !== 3");
    assert(v === c, "class === the same class");
    assert(!(v === new C()), "class === another class");
}

function stringNumberOrNull() {
    let u: string | number | null = "a";
    assert(u === "a", "string === a");
    assert(!(u === "b"), "string === b");
    assert(!(u === 1), "string === 1");

    u = null;
    assert(!(u === "a"), "null === a");
    assert(!(u === 1), "null === 1");

    u = 1;
    assert(u === 1, "number === 1");
    assert(!(u === "a"), "number === a");
}

// narrowed by typeof, `string | number | null` is left `string | null`: with strict null checks that is
// the string's own pointer, not a union with a tag, and the value is read out of the tagged one
function afterTypeOf(u: string | number | null) {
    if (typeof u == "number") {
        return 1;
    }

    return u === "a" ? 2 : 3;
}

function main() {
    nullableNumber();
    classOrNumber();
    stringNumberOrNull();
    assert(afterTypeOf(5) == 1, "narrowed: number");
    assert(afterTypeOf("a") == 2, "narrowed: a");
    assert(afterTypeOf("b") == 3, "narrowed: b");
    assert(afterTypeOf(null) == 3, "narrowed: null");
    print("done.");
}
