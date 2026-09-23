import './export_throw_class'

// Catching a class that only another module throws.
// - Linux: an untyped catch finds out what it caught from the exception's type_info, and only
//   knew the classes its own module throws - so this class was boxed as a plain object and
//   `<M.Failure>e` threw an invalid cast. The class's type_info now carries a box thunk.
// - -shared: `instanceof` and `<M.Failure>e` of a class imported from a shared library dropped
//   the instanceOf call (the rtti value was expected to be a ref), so the first did not compile
//   and the second always threw.
// - static link, Windows: a typed catch and the throw each emitted the class's box thunk, and
//   the two collided (duplicate symbol).

function typedCatch() {
    let r: number = 0;
    try {
        M.fail(1);
    } catch (e: M.Failure) {
        r = e.code;
    }

    return r;
}

function untypedCatch() {
    let r: number = 0;
    try {
        M.fail(2);
    } catch (e) {
        r = (<M.Failure>e).code;
    }

    return r;
}

function anyCatch() {
    let r: number = 0;
    try {
        M.fail(3);
    } catch (e: any) {
        r = (<M.Failure>e).code;
    }

    return r;
}

function instanceofCatch() {
    let r: number = 0;
    try {
        M.fail(4);
    } catch (e: any) {
        if (e instanceof M.Failure) {
            r = e.code;
        }
    }

    return r;
}

function main() {
    assert(typedCatch() == 1, "catch (e: M.Failure) of a class thrown in another module");
    assert(untypedCatch() == 2, "<M.Failure>e of a class thrown in another module");
    assert(anyCatch() == 3, "catch (e: any) of a class thrown in another module");
    assert(instanceofCatch() == 4, "instanceof on a caught class thrown in another module");

    print("done.");
}
