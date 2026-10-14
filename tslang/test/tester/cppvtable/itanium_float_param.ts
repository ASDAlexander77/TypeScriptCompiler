// Compiled for x86-64 Linux: a record of two floats passed by value goes in SSE registers, which
// isn't done; it is an error, not a wrong call.
type Pair = [first: f32, second: f32];

@cpp
class Methods {
    @slot(2) take(p: Pair): void;
}

function call(methods: Methods): void {
    methods.take([1, 2]);
}

function main() {
    print("done.");
}
