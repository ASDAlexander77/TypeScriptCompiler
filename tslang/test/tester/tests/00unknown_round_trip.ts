// `unknown` holds what `any` does, the same box: a value put into it is boxed as into `any` and read
// out of it as out of `any`, and `typeof` asks the box. A class object put into `unknown` was the
// address of a stack copy of its pointer, which the cast back read as the object (`(<W>o).n` read
// garbage); a number put into it pointed into the caller's frame.
class W {
    constructor(public n: number) {}
}

function field(o: unknown) {
    return (<W>o).n;
}

function kind(o: unknown) {
    return typeof o;
}

function kindAny(o: any) {
    return typeof o;
}

function num(o: unknown) {
    return <number>o;
}

function str(o: unknown) {
    return <string>o;
}

function keep(o: unknown): unknown {
    return o;
}

function main() {
    assert(field(new W(5)) == 5);
    assert(field(keep(new W(6))) == 6);

    const n: number = 3.5;
    assert(kind(n) == "number");
    assert(kind("s") == "string");
    assert(kind(new W(1)) == kindAny(new W(1)));

    assert(num(n) == 3.5);
    assert(str("hi") == "hi");

    let u: unknown = n;
    assert(typeof u == "number");
    u = "x";
    assert(typeof u == "string");
    assert(<string>u == "x");

    print("done.");
}
