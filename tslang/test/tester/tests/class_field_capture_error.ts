// A field initializer of a class declared in a function, without a constructor of its own, that reads
// one of the function's locals: the class can't capture yet, and the error names the variable.
function make(start: number) {
    class Counter {
        value = start;
    }

    return new Counter().value;
}

function main() {
    print(make(1));
    print("done.");
}
