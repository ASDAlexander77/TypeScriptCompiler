// -mm=own: a value that owns memory returned through a finally block that yields is an error at the return (the
// value kept for after the finally is a field of the generator, and the return would hand out a borrow of it).
function* g(n: number) {
    try {
        yield "1";
        return "r" + n;
    } finally {
        yield "2";
    }
}

function main() {
    for (const v of g(1)) print(v);
}
