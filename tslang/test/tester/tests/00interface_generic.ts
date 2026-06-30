interface I<T> {
    foo(x: number): T;
}

interface A extends I<number>/*, I<string>*/ { }

class AI {
    foo(x: number) { return x; }
}

function main() {
    let x: A = new AI();
    const r = x.foo(1); // no error
    //const r2 = x.foo(''); // error

    assert(r == 1);

    let x2: I<number> = new AI();
    const r2 = x2.foo(2); // no error

    assert(r2 == 2);

    print("done.");
}