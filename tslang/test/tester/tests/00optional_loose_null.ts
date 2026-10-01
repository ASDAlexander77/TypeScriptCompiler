// Loosely, null and undefined are equal: an optional that holds null is `== undefined` as well as
// `== null`, and one that holds undefined is both too
function nullableString() {
    let s: string | null | undefined = null;
    assert(s == undefined, "string null == undefined");
    assert(!(s != undefined), "string null != undefined");
    assert(s == null, "string null == null");
    assert(!(s === undefined), "string null === undefined");

    s = undefined;
    assert(s == undefined, "string undefined == undefined");
    assert(s == null, "string undefined == null");

    s = "a";
    assert(!(s == undefined), "string value == undefined");
    assert(s != undefined, "string value != undefined");
}

function nullableNumber() {
    let n: number | null | undefined = null;
    assert(n == undefined, "number null == undefined");
    assert(!(n != undefined), "number null != undefined");
    assert(!(n === undefined), "number null === undefined");

    n = 1;
    assert(!(n == undefined), "number value == undefined");
}

class C {
    x = 1;
}

function nullableClass() {
    let c: C | null | undefined = null;
    assert(c == undefined, "class null == undefined");

    c = new C();
    assert(!(c == undefined), "class value == undefined");
}

function main() {
    nullableString();
    nullableNumber();
    nullableClass();
    print("done.");
}
