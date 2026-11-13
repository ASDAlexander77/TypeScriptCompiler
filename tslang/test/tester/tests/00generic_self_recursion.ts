// An instance of a generic function calls itself: countDown<string> from its own body. Its declared
// return type is its type before its body is known, so the call has a result to use. It failed with
// "not all generic types can be identified" - only a call whose result was not used compiled.

function countDown<T>(value: T, n: number): number {
    if (n > 0) {
        return countDown(value, n - 1) + 1;
    }

    return 0;
}

function countDownTyped<T>(value: T, n: number): number {
    return n > 0 ? countDownTyped<T>(value, n - 1) + 1 : 0;
}

class Counter {
    count<T>(value: T, n: number): number {
        return n > 0 ? this.count(value, n - 1) + 1 : 0;
    }
}

function isEven<T>(value: T, n: number): boolean {
    return n == 0 ? true : isOdd(value, n - 1);
}

function isOdd<T>(value: T, n: number): boolean {
    return n == 0 ? false : isEven(value, n - 1);
}

function last<T>(items: T[], i: number): T {
    return i + 1 < items.length ? last(items, i + 1) : items[i];
}

function reversed<T>(items: T[], i: number): T[] {
    if (i < 0) {
        return [];
    }

    const rest = reversed(items, i - 1);
    return [items[i], ...rest];
}

function main() {
    assert(countDown("x", 5) == 5, "a call of itself, its result used");
    assert(countDown(1, 2) == 2, "another instance");
    assert(countDownTyped(true, 3) == 3, "with its type argument");
    assert(new Counter().count("x", 4) == 4, "a generic method");
    assert(isEven(1, 10) && isOdd("a", 7), "two that call each other");
    assert(last([1, 2, 3], 0) == 3 && last(["a", "b"], 0) == "b", "returning T");
    const back = reversed([1, 2, 3], 2);
    assert(back.length == 3 && back[0] == 3 && back[2] == 1, "returning T[]");
    print("done.");
}
