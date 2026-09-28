// A and B are unrelated and their `v` fields have different types; the cast compiled, and reading
// b.v read a number as a string.
class A {
    constructor(public v: number) {}
}

class B {
    constructor(public v: string) {}
}

function main() {
    const a = new A(1);
    const b: B = a;
    print(b.v);
}
