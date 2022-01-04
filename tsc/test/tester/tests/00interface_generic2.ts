interface I<T> {
    foo(x: number): T;
}

class AI {
    foo(x: number) { return x; }
}

function main() {
    let x: I<number> = new AI();
    const r = x.foo(1); // no error

    print("done.");
}