class S {
    a = 30.0;
    b = "ddd";
    r = true;
}

interface IS {
    a: number;
    b: string;
    r: boolean;
}

function main() {
    let o = { a: 10, b: "asd" };

    assert(o.a == 10)
    assert(o.b == "asd")

    o = { s: "xxx", a: 20, b: "sss", c: "ddd" };

    assert(o.a == 20)
    assert(o.b == "sss")

    o = new S();

    print(o.a, o.b);

    assert(o.a == 30)
    assert(o.b == "ddd")

    o = <IS>new S();

    print(o.a, o.b);

    assert(o.a == 30)
    assert(o.b == "ddd")

    print("done.");
}
