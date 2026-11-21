// A yield in a finally block of a generator is an error at the yield: the finally is copied for each way out of
// the try (one of them inside an exception handler) and a resume landed in one copy only - the yield was skipped
// and the rest of the finally ran from the wrong place.
function* g() {
    try {
        yield 1;
    } finally {
        print("in finally");
        yield 2;
    }
}

function main() {
    for (const v of g()) print(v);
}
