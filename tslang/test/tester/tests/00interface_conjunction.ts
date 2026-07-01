interface F1 {
    a: number;
    a2: boolean;
}

interface F2 {
    b: string;
    b2: number;
}

type t = F1 & F2 & { c: number };

interface t2 extends F1, F2 {
    c: number;
}

type tt = { a: boolean };
type tt2 = { b: number };
type tt3 = { c: string };

type r = tt & tt2 & tt3;

function main() {

    const f1: F1 = { a: 10.0, a2: true };
    print(f1.a, f1.a2);

    assert(f1.a2);

    const f2: F2 = { b: "Hello1", b2: 20.0 };
    print(f2.b, f2.b2);

    assert(f2.b2 == 20.0);

    const a: t = { a: 10.0, a2: true, b: "Hello", b2: 20.0, c: 30.0 };
    print(a.a, a.a2, a.b, a.b2);

    assert(a.a2);
    assert(a.b2 == 20.0);
    assert(a.c == 30.0);

    const b: t2 = { a: 10.0, a2: true, b: "Hello", b2: 20.0, c: 30.0 };
    print(b.a, b.a2, b.b, b.b2, b.c);

    assert(b.a2);
    assert(b.b2 == 20.0);
    assert(b.c == 30.0);

    const c: r = { a: true, b: 10.0, c: "Hello" };
    print(c.a, c.b, c.c);

    assert(c.a);
    assert(c.b == 10.0);
    assert(c.c == "Hello");

    print("done.");
}
