namespace Array {
    function of<T>(...arg: T[]) {
        return arg;
    }
}

function main() {
    const arr = Array.of('foo', <number>2, 'bar', true);

    for (const a of arr) {
        print("item :");
        if (typeof a == "string") {
            print("str:", a);
        }
        if (typeof a == "number") {
            print("str:", a);
            assert(a == 2);
        }
        if (typeof a == "boolean") {
            print("bool:", a);
            assert(a);
        }
    }

    print("done.");
}