// regression test: TryOpLowering's typed-catch RTTI compare (Itanium/Linux
// exception path) used to fall through to normal post-try control flow when
// the caught value's type didn't match the catch clause, instead of
// rethrowing. This silently swallowed exceptions of the wrong type.
//
// Nested try/catch here forces a real mismatch-then-rethrow: the inner catch
// only accepts a number, so a thrown string must propagate past it and be
// caught by the outer string handler instead of being (wrongly) caught, or
// silently falling through with the try's normal result.

function inner(): string {
    try {
        throw "boom";
    } catch (e: TypeOf<1>) {
        return "WRONGLY CAUGHT BY INNER";
    }
}

function testMismatchRethrow(): string {
    try {
        return inner();
    } catch (e: string) {
        return "correctly rethrown and caught outer: " + e;
    }
}

function main() {
    let result = testMismatchRethrow();
    print(result);
    assert(result == "correctly rethrown and caught outer: boom", "mismatched catch must rethrow, not swallow");

    print("done.");
}
