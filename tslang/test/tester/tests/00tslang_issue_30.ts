// TSLANG_ISSUES 30: a generic is instantiated where it is first used, and a name in its body resolved
// to a local of the function using it ("not a function to call" for `total(...)` when the caller had
// `const total = 5`). A generic declared outside every function sees no local of another function; one
// declared inside a function still reads that function's locals, and a lambda in a generic still
// captures the generic's own.

function total(items: number[]) {
    let sum: number = 0;
    for (const item of items) sum += item;
    return sum;
}

let scale = 10;

class Bag<T> {
    constructor(public items: number[]) {}

    sum() {
        return total(this.items);
    }

    scaled() {
        return total(this.items) * scale;
    }

    // a lambda capturing a local of the generic's own method
    adder() {
        const offset = 100;
        return (x: number) => x + offset;
    }
}

function sumOf<T>(items: number[]) {
    return total(items);
}

function scaledBy<T>(value: number) {
    return value * scale;
}

// a generic method of a plain class, and a generic arrow in a module const
class Plain {
    sum<T>(items: number[]) {
        return total(items);
    }
}

const arrowSum = <T>(items: number[]) => total(items);

function main() {
    // locals named like what the generics use
    const total = 5;
    const scale = 2;
    const offset = -1;

    const bag = new Bag<int>([1, 2, 3]);
    assert(bag.sum() == 6, "a generic class method calls the module's function");
    assert(bag.scaled() == 60, "and reads the module's variable");
    assert(bag.adder()(1) == 101, "a lambda in a generic captures the generic's local");
    assert(sumOf<int>([4, 5]) == 9, "a generic function calls the module's function");
    assert(scaledBy<int>(3) == 30, "a generic function reads the module's variable");
    assert(new Plain().sum<int>([5, 6]) == 11, "a generic method");
    assert(arrowSum<int>([7, 8]) == 15, "a generic arrow");
    assert(new Late<int>([1, 2]).sum() == 3, "a generic class declared after the caller");
    assert(lateSum<int>([3, 4]) == 7, "a generic function declared after the caller");
    assert(total == 5 && scale == 2 && offset == -1, "the caller's locals are its own");

    // a generic declared inside a function reads that function's locals
    const factor = 7;
    function times<T>(value: number) {
        return value * factor;
    }
    assert(times<int>(3) == 21, "a local generic reads its function's local");

    print("done.");
}

// declared after main, which instantiates them
class Late<T> {
    constructor(public items: number[]) {}

    sum() {
        return total(this.items);
    }
}

function lateSum<T>(items: number[]) {
    return total(items);
}
