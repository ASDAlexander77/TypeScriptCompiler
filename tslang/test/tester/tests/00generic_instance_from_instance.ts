// A generic function's instance can make another instance of the same function while it is being made:
// `flatten<number[][]>` calls `flatten<number[]>`, which calls `flatten<number>`. This failed with
// "not all generic types can be identified" unless the inner instances had been made before, which is how
// `JSON.stringify([[1], [2]])` failed in a program that had not stringified a number first.

function flatten<T>(value: T): string {
    if (typeof value == "array") {
        let result = "";
        for (const element of value) {
            result += flatten(element);
        }

        return result;
    } else {
        return "<" + value + ">";
    }
}

// two generic functions that call each other's instances, with a default type parameter
function wrap<T, R = string>(value: T): R {
    if (typeof value == "array") {
        return "[" + each(value) + "]";
    } else {
        return "" + value;
    }
}

function each<T>(values: T[]): string {
    let result = "";
    for (const value of values) {
        result += wrap(value);
    }

    return result;
}

function main() {
    assert(flatten([[1], [2, 3]]) == "<1><2><3>", "an instance made from an instance");
    assert(flatten([[["a"]], [["b", "c"]]]) == "<a><b><c>", "three deep");
    assert(wrap([[1, 2], [3]]) == "[[12][3]]", "two functions, each making the other's instances");
    print("done.");
}
