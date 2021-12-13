interface F1 {
    a: number;
    a2: boolean;
}

interface F2 {
    b: string;
    b2: number;
}

type t = F1 & F2;

interface t2 extends F1, F2 {
}

function main() {

    const f1: F1 = { a: 10.0, a2: true };
    print(f1.a, f1.a2);

    assert(f1.a2);

    const f2: F2 = { b: "Hello1", b2: 20.0 };
    print(f2.b, f2.b2);

    assert(f2.b2 == 20.0);

    const a: t = { a: 10.0, a2: true, b: "Hello", b2: 20.0 };
    print(a.a, a.a2, a.b, a.b2);

    assert(a.a2);
    assert(a.b2 == 20.0);

    const b: t2 = { a: 10.0, a2: true, b: "Hello", b2: 20.0 };
    print(b.a, b.a2, b.b, b.b2);

    assert(b.a2);
    assert(b.b2 == 20.0);

    print("done.");
}
