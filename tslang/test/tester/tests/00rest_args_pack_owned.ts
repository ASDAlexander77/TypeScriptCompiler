// The array a rest parameter receives is built at the call from the arguments, and like an array
// literal's it releases every element when it dies - so it has to take a reference to each. Under rc
// a call through an extension function, `fruits.concat(["x", "y"])`, packed its arguments without
// one: the callee's first `for (const item of other)` took the only reference and gave it back,
// freeing the array, and the second loop read what was left (the default library's Array.concat,
// which printed garbage for the first element it copied and corrupted the heap).

namespace __Array {
    function joinedWith<T>(this: T[], ...other: T[][]): string {
        let count = this.length;
        for (const item of other) {
            count += item.length;
        }

        let result = "";
        for (const v of this) {
            result += v;
        }

        for (const item of other) {
            for (const v of item) {
                result += v;
            }
        }

        return result + count;
    }
}

class Vec {
    x: number;

    constructor(x: number) {
        this.x = x;
    }
}

// Allocate over whatever has just been freed, so a use-after-free reads something else.
function churn() {
    for (let i = 0; i < 64; i++) {
        let filler = new Vec(999);
    }
}

function viaExtension() {
    const fruits = ["a", "b"];
    return fruits.joinedWith(["x", "y"], ["z"]);
}

// a plain function's rest parameter, the elements made at the call
function lengths(...items: string[][]): int {
    let n = 0;
    for (const item of items) {
        n += item.length;
    }

    churn();
    for (const item of items) {
        n += item.length;
    }

    return n;
}

function main() {
    assert(viaExtension() == "abxyz5", "a rest parameter's array holds its elements through an extension call");
    assert(lengths(["p", "q"], ["r" + 1]) == 6, "a rest parameter's array holds elements made at the call");

    print("done.");
}
