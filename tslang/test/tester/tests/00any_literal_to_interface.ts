// `any` to an interface of an object literal boxed as it is throws: the box says only that it holds an
// object literal ('object', or 'tuple' for one of fields only), not which fields, so no vtable can be
// laid out for it. Its block was read as the interface's (this, vtable) pair, and the call through it
// crashed, under every memory model. An interface value made of a literal before it was boxed still
// round-trips. A boxed null gives the null interface: under rc, none and own it was a box of its own,
// read as a pair that was not null.
interface Adder {
    add(x: number): number;
}

function asAdder(x: any) {
    return <Adder>x;
}

function throws(x: any) {
    let thrown = false;
    try {
        asAdder(x);
    } catch (e) {
        thrown = true;
    }

    return thrown;
}

function main() {
    const fields = { add: (x: number) => x + 10 };
    assert(throws(fields), "a literal of fields only");

    const methods = { add(x: number) { return x + 20; } };
    assert(throws(methods), "a literal with a method");

    const i: Adder = { add(x: number) { return x + 30; } };
    assert(asAdder(i).add(1) == 31, "an interface value round-trips");

    assert(asAdder(null) == null, "null");

    print("done.");
}
