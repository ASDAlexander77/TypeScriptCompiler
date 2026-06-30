type int = TypeOf<1>;

function f(a: [int, string][]) {
    for (const [k, v] of a) {
        print(k, v);
    }
}

function main() {
    for (const q of [1, 12]) {
        print(q);
    }

    for (const l of "Hello") {
        print(l);
    }

    const array1 = ["a", "b", "c"];

    for (const element of array1) {
        print(element);
    }

    const trees = [
        [1, "redwood"],
        [2, "bay"],
        [3, "cedar"],
        [4, "oak"],
        [5, "maple"],
    ];

    for (const [k, v] of trees) {
        print(k, v);
    }

    f(trees);

    print("done.");
}
