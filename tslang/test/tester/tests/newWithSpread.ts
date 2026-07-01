let count = 0;

interface A {
    f: {
        new(x: number, y: number, ...z: string[]);
    }
}

// TODO: fix code which is commented 
class B {
    constructor(x: number, y: number, ...z: string[]) { print("constr", x, y); print("z length:", z.length); for (const v of z) print("z[...]: ", v); count++; }
}

interface C {
    "a-b": typeof B;
}

interface D {
    1: typeof B;
}

function main() {

    let a: string[] = [];

    let weakB: {
        new(x: number, y: number, ...z: string[]): B;
    } = B;

    const b0 = new weakB(1, 2, "string");
    const b1 = new weakB(1, 2, ...a);
    const b2 = new weakB(1, 2, ...a, "string");

    let b: A = { f: B };
    let c: C;
    let d: A[] = [b, b];
    let g: C[];

    // Property access expression
    new b.f(1, 2, "string");
    new b.f(1, 2, ...a);
    new b.f(1, 2, ...a, "string");

    // Parenthesised expression
    new (b.f)(1, 2, "string");
    new (b.f)(1, 2, ...a);
    new (b.f)(1, 2, ...a, "string");

    // Element access expression
    new d[1].f(1, 2, "string");
    new d[1].f(1, 2, ...a);
    new d[1].f(1, 2, ...a, "string");

    // Basic expression
    new B(1, 2, "string");
    new B(1, 2, ...a);
    new B(1, 2, ...a, "string");

    // Property access expression
    new c["a-b"](1, 2, "string");
    new c["a-b"](1, 2, ...a);
    new c["a-b"](1, 2, ...a, "string");

    // Parenthesised expression
    new (c["a-b"])(1, 2, "string");
    new (c["a-b"])(1, 2, ...a);
    new (c["a-b"])(1, 2, ...a, "string");

    // Element access expression
    new g[1]["a-b"](1, 2, "string");
    new g[1]["a-b"](1, 2, ...a);
    new g[1]["a-b"](1, 2, ...a, "string");

    print(count);

    assert(count == 24);

    print("done.");
}