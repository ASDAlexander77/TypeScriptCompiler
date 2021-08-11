type int = 1;
type const_string = "";

function main() {
    const a = 1;
    const aAny = <any>a;
    const b = <int>aAny;

    print(typeof a);
    print(typeof aAny);
    print(typeof b);

    print(a, b);

    assert(a == b);

    assert(typeof a == typeof aAny);
    assert(typeof aAny == typeof b);
    assert(typeof a == typeof b);

    const s = "string value";
    const sAny = <any>s;
    const ss = <const_string>sAny;

    print(typeof s);
    print(typeof sAny);
    print(typeof ss);

    print(s, ss);

    assert(s == ss);

    assert(typeof s == typeof sAny);
    assert(typeof sAny == typeof ss);
    assert(typeof s == typeof ss);

    const s2 = "string value";
    const s2Any = <any>s2;
    const ss2 = <string>s2Any;

    print(typeof s2);
    print(typeof s2Any);
    print(typeof ss2);

    print(s2, ss2);

    assert(s2 == ss2);

    assert(typeof s2 == typeof s2Any);
    assert(typeof s2Any == typeof ss2);
    assert(typeof s2 == typeof ss2);

    print("done.");
}
