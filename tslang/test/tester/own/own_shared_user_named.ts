// A program's own class named Shared wins over the built-in (spec 23.1).
class Shared {
    constructor(public value: number) {}
    static count(s: Shared) {
        return 42;
    }
}

function main() {
    const x = new Shared(1);
    assert(x.value == 1, "the program's own Shared");
    assert(Shared.count(x) == 42, "the program's own static count");
    print("done.");
}
