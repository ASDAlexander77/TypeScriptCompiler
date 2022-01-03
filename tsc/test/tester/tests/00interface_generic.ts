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

    print("done.");
}