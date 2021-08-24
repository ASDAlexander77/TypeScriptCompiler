class IntIter {
    constructor(private i = 0) {}
    next() {
        type retType = [value: typeof this.i, done: boolean];
        if (this.i < 10) {
            return <retType>[this.i++, false];
        }

        return <retType>[this.i, true];
    }
}

function main() {
    let it = new IntIter();

    let count = 0;
    for (const o of it) {
        count++;
        print(o);
    }

    assert(count == 10);

    count = 0;
    for (const o of "Hello") {
        count++;
        print(o);
    }

    assert(count == 5);

    print("done.");
}
