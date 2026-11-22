// -mm=own: an object thrown through a finally block that yields is an error at the throw (the exception is kept in an
// `any` for after the finally, and an `any` holding an object can't be copied there).
class Failure {
    constructor(public message: string) {}
}

function* g(n: number) {
    try {
        yield 1;
        if (n > 0) throw new Failure("f" + n);
    } finally {
        yield 2;
    }
}

function main() {
    for (const v of g(1)) print(v);
}
