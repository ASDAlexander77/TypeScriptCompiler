class StringIterator {
    next() {
        print("next...");

        return {
            done: true,
            value: ""
        };
    }
    [Symbol.iterator]() {
        // TODO: finish it, should be called
        print("iterator...");
        return this;
    }
}

function main() {
    for (const v of new StringIterator) { }

    print("done.");
}