// A yield in a catch block of a generator is an error at the yield: the generator can't resume inside the
// block's exception handler (the compiler crashed in LLVM's instruction selection).
function* g() {
    try {
        yield 1;
        throw "e";
    } catch (e) {
        yield 2;
    }
}

function main() {
    for (const v of g()) print(v);
}
