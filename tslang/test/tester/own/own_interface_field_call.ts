// A function-typed field of an interface (`toString: () => string`), called: MLIRGen binds the
// interface's `this` to the function it read out of the field and splits the bound function straight
// back into `this` and the function for the call. The call borrows the object, as a method call does.
interface Formattable {
    prefix: string;
    toString: () => string;
}

function show(f: Formattable) {
    return f.toString();
}

function main() {
    const a = {
        prefix: "A: ",
        toString() {
            return this.prefix + "hello";
        },
    };

    const iface: Formattable = a;
    print(iface.toString());
    assert(iface.toString() == "A: hello");
    assert(show(iface) == "A: hello");

    print("done.");
}
